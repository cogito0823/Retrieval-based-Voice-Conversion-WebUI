import os
import sys
import shutil
import threading
import time
import uuid
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[3]  # repo root

# Make repo root importable (so `configs`, `infer`, etc. resolve)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.config import Config
from infer.modules.vc.modules import VC

# Controlled directories (v1: scan-only, no model upload API yet)
WEIGHTS_DIR = Path(os.getenv("RVCV2_WEIGHT_DIR", str(ROOT_DIR / "assets" / "weights"))).resolve()
INDICES_DIR = Path(os.getenv("RVCV2_INDEX_DIR", str(ROOT_DIR / "assets" / "indices"))).resolve()
# Logs directory contains training outputs including .index files (matches old WebUI behavior)
LOGS_DIR = Path(os.getenv("RVCV2_LOGS_DIR", str(ROOT_DIR / "logs"))).resolve()

# Server storage (uploads, artifacts, tasks)
STORAGE_DIR = Path(os.getenv("RVCV2_STORAGE_DIR", str(ROOT_DIR / "web" / "server" / "storage"))).resolve()
UPLOADS_DIR = STORAGE_DIR / "uploads"
ARTIFACTS_DIR = STORAGE_DIR / "artifacts"
TASKS_DIR = STORAGE_DIR / "tasks"


def _ensure_dirs() -> None:
    for p in (UPLOADS_DIR, ARTIFACTS_DIR, TASKS_DIR):
        p.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str) -> str:
    # Very small sanitizer: keep basename only.
    return os.path.basename(name).strip() or "file"


def _uuid() -> str:
    return uuid.uuid4().hex


class ModelItem(BaseModel):
    id: str
    name: str


class IndexItem(BaseModel):
    id: str
    name: str


class UploadResp(BaseModel):
    uploadId: str
    filename: str


TaskStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class TaskInfo(BaseModel):
    id: str
    type: str = "infer_offline"
    status: TaskStatus
    progress: float = Field(0, ge=0, le=100)
    createdAt: float
    startedAt: float | None = None
    finishedAt: float | None = None
    resultArtifactId: str | None = None
    error: str | None = None


class CreateInferOfflineReq(BaseModel):
    modelId: str
    audioUploadId: str
    indexId: str | None = None

    f0_up_key: int = 0
    f0_method: str = "rmvpe"
    index_rate: float = Field(0.88, ge=0, le=1)
    filter_radius: int = Field(3, ge=0, le=7)
    resample_sr: int = Field(0, ge=0, le=48000)
    rms_mix_rate: float = Field(1.0, ge=0, le=1)
    protect: float = Field(0.33, ge=0, le=0.5)
    f0UploadId: str | None = None


class CreateTaskResp(BaseModel):
    taskId: str


class LogsResp(BaseModel):
    items: list[dict[str, Any]]
    nextCursor: str | None = None


class InferDefaults(BaseModel):
    f0_up_key: int = 0
    f0_method: str = "rmvpe"
    index_rate: float = 0.75
    filter_radius: int = 3
    resample_sr: int = 0
    rms_mix_rate: float = 0.25
    protect: float = 0.33


class InferOptionsResp(BaseModel):
    f0_methods: list[str]
    resample_srs: list[int]
    defaults: InferDefaults


@dataclass
class TaskRecord:
    info: TaskInfo
    req: CreateInferOfflineReq
    log_path: Path
    cancel: bool = False
    proc: subprocess.Popen | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class TaskManager:
    """
    v1: in-memory task registry + single worker thread (serial execution).
    Logs are persisted to JSONL on disk.
    """

    def __init__(self) -> None:
        self._q: "Queue[str]" = Queue()
        self._tasks: dict[str, TaskRecord] = {}
        self._worker: threading.Thread | None = None

        # NOTE:
        # `configs.config.Config` parses CLI args from sys.argv at init time.
        # When running under `uvicorn`, sys.argv contains uvicorn arguments,
        # which would crash with "unrecognized arguments".
        # For server/library usage here, we temporarily sanitize argv.
        _argv = sys.argv[:]
        sys.argv = [_argv[0]]
        try:
            self._config = Config()
        finally:
            sys.argv = _argv
        self._vc = VC(self._config)
        self._loaded_model_id: str | None = None

        # Ensure VC uses our controlled dirs when it reads env vars.
        os.environ.setdefault("weight_root", str(WEIGHTS_DIR))
        os.environ.setdefault("index_root", str(INDICES_DIR))
        # Required by rmvpe f0 method (infer/modules/vc/pipeline.py reads os.environ["rmvpe_root"])
        os.environ.setdefault("rmvpe_root", str((ROOT_DIR / "assets" / "rmvpe").resolve()))
        # Commonly used by HuBERT loader in this repo (safe default)
        os.environ.setdefault("hubert_root", str((ROOT_DIR / "assets" / "hubert").resolve()))

    def start(self) -> None:
        if self._worker is not None:
            return
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def create_infer_task(self, req: CreateInferOfflineReq) -> str:
        task_id = _uuid()
        now = time.time()
        log_dir = TASKS_DIR / task_id
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "events.jsonl"
        info = TaskInfo(id=task_id, status="queued", createdAt=now)
        rec = TaskRecord(info=info, req=req, log_path=log_path)
        self._tasks[task_id] = rec
        self._append_log(task_id, {"ts": now, "kind": "status", "status": "queued"})
        self._q.put(task_id)
        return task_id

    def get(self, task_id: str) -> TaskInfo:
        rec = self._tasks.get(task_id)
        if not rec:
            raise KeyError(task_id)
        return rec.info

    def cancel(self, task_id: str) -> TaskInfo:
        rec = self._tasks.get(task_id)
        if not rec:
            raise KeyError(task_id)
        rec.cancel = True
        # If running with a subprocess, terminate it best-effort.
        if rec.proc and rec.proc.poll() is None:
            try:
                rec.proc.terminate()
            except Exception:
                pass
        # If not started yet, it will be marked cancelled when dequeued.
        if rec.info.status in ("queued", "running"):
            self._set_status(task_id, "cancelled")
        return rec.info

    def read_logs(self, task_id: str, cursor: str | None, limit: int = 200) -> LogsResp:
        rec = self._tasks.get(task_id)
        if not rec:
            raise KeyError(task_id)
        if not rec.log_path.exists():
            return LogsResp(items=[], nextCursor=None)

        start = int(cursor) if cursor else 0
        items: list[dict[str, Any]] = []
        next_cursor: str | None = None

        with rec.log_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < start:
                    continue
                if len(items) >= limit:
                    next_cursor = str(i)
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    import json

                    items.append(json.loads(line))
                except Exception:
                    items.append({"ts": time.time(), "kind": "raw", "text": line})

        return LogsResp(items=items, nextCursor=next_cursor)

    def _append_log(self, task_id: str, obj: dict[str, Any]) -> None:
        rec = self._tasks.get(task_id)
        if not rec:
            return
        import json

        with rec.lock:
            with rec.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _set_status(self, task_id: str, status: TaskStatus, **kwargs: Any) -> None:
        rec = self._tasks.get(task_id)
        if not rec:
            return
        now = time.time()
        rec.info.status = status
        if status == "running":
            rec.info.startedAt = now
        if status in ("succeeded", "failed", "cancelled"):
            rec.info.finishedAt = now
        for k, v in kwargs.items():
            if hasattr(rec.info, k):
                setattr(rec.info, k, v)
        self._append_log(task_id, {"ts": now, "kind": "status", "status": status, **kwargs})

    def _run(self) -> None:
        while True:
            task_id = self._q.get()
            rec = self._tasks.get(task_id)
            if not rec:
                continue
            if rec.cancel:
                self._set_status(task_id, "cancelled")
                continue
            self._execute_infer_offline_subprocess(task_id, rec)

    def _execute_infer_offline_subprocess(self, task_id: str, rec: TaskRecord) -> None:
        self._set_status(task_id, "running", progress=0.0)
        req = rec.req
        try:
            audio_path = _resolve_upload_path(req.audioUploadId)
            if not audio_path.exists():
                raise RuntimeError("audioUploadId not found")

            # Resolve index path (optional)
            index_real = _resolve_index_path(req.indexId) if req.indexId else None

            # Build worker args (avoid in-process hangs; allow terminate)
            worker_py = ROOT_DIR / "web" / "server" / "app" / "infer_worker.py"
            if not worker_py.exists():
                raise RuntimeError("infer worker script missing")

            env = os.environ.copy()
            env.setdefault("weight_root", str(WEIGHTS_DIR))
            # Use LOGS_DIR as index_root (matches old WebUI behavior: index_root=logs)
            env.setdefault("index_root", str(LOGS_DIR))
            env.setdefault("rmvpe_root", str((ROOT_DIR / "assets" / "rmvpe").resolve()))
            env.setdefault("hubert_root", str((ROOT_DIR / "assets" / "hubert").resolve()))

            cmd = [
                sys.executable,
                str(worker_py),
                "--task-id",
                task_id,
                "--model-id",
                req.modelId,
                "--audio-path",
                str(audio_path),
                "--f0-up-key",
                str(req.f0_up_key),
                "--f0-method",
                req.f0_method,
                "--index-rate",
                str(req.index_rate),
                "--filter-radius",
                str(req.filter_radius),
                "--resample-sr",
                str(req.resample_sr),
                "--rms-mix-rate",
                str(req.rms_mix_rate),
                "--protect",
                str(req.protect),
                "--artifacts-dir",
                str(ARTIFACTS_DIR),
            ]
            if index_real:
                cmd += ["--index-path", str(index_real)]
            if req.f0UploadId:
                f0p = _resolve_upload_path(req.f0UploadId)
                if f0p.exists():
                    cmd += ["--f0-file", str(f0p)]

            p = subprocess.Popen(
                cmd,
                cwd=str(ROOT_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            rec.proc = p

            # watchdogs:
            # - `last_change_ts`: only updates when progress meaningfully changes (pct/text changes)
            # - if stuck at same stage (vc_single heartbeat repeats), we will NOT update it,
            #   and eventually terminate worker to avoid infinite hang.
            last_change_ts = time.time()
            last_pct: float | None = None
            last_text: str = ""
            last_line: str = ""
            while True:
                if rec.cancel:
                    try:
                        p.terminate()
                    except Exception:
                        pass
                    self._set_status(task_id, "cancelled")
                    return

                line = p.stdout.readline() if p.stdout else ""
                if not line:
                    if p.poll() is not None:
                        break
                    # watchdog: if no meaningful progress update for too long, fail and kill
                    if time.time() - last_change_ts > 60 and rec.info.status == "running":
                        try:
                            p.terminate()
                        except Exception:
                            pass
                        self._set_status(
                            task_id,
                            "failed",
                            error=(
                                "推理长时间停留在同一阶段，疑似卡死（常见于 macOS MPS/某些推理算子）。"
                                "建议尝试：切换音高算法为 pm/harvest，或后续增加“强制 CPU 推理”开关。"
                            ),
                        )
                        return
                    time.sleep(0.2)
                    continue

                line = line.strip("\n")
                if not line:
                    continue
                last_line = line
                # worker prints JSON per line
                try:
                    import json

                    obj = json.loads(line)
                except Exception:
                    obj = {"kind": "log", "text": line}
                kind = obj.get("kind", "log")
                if kind == "progress":
                    text = str(obj.get("text", ""))
                    pct = obj.get("pct")
                    pct_f: float | None = None
                    if isinstance(pct, (int, float)):
                        pct_f = float(pct)
                        rec.info.progress = pct_f

                    # Only treat as progress if changed (avoid vc_single heartbeat spam)
                    changed = False
                    if pct_f is not None and (last_pct is None or abs(pct_f - last_pct) >= 0.1):
                        changed = True
                    if text and text != last_text:
                        changed = True
                    if changed:
                        self._append_log(task_id, {"ts": time.time(), "kind": "log", "text": text})
                        last_change_ts = time.time()
                        if pct_f is not None:
                            last_pct = pct_f
                        last_text = text
                elif kind == "done":
                    success = bool(obj.get("success"))
                    if success:
                        artifact_id = str(obj.get("artifactId", "")) or _uuid()
                        rec.info.progress = 100.0
                        self._set_status(task_id, "succeeded", progress=100.0, resultArtifactId=artifact_id)
                    else:
                        err = str(obj.get("error", "")) or "inference failed"
                        self._set_status(task_id, "failed", error=err)
                    return
                else:
                    text = str(obj.get("text", "")) or line
                    pct = _extract_pct(text)
                    changed = False
                    if pct is not None and (last_pct is None or abs(pct - last_pct) >= 0.1):
                        rec.info.progress = pct
                        last_pct = pct
                        changed = True
                    if text and text != last_text:
                        changed = True
                    if changed:
                        self._append_log(task_id, {"ts": time.time(), "kind": "log", "text": text})
                        last_text = text
                        last_change_ts = time.time()

            # process ended without done
            if rec.cancel or rec.info.status == "cancelled":
                self._set_status(task_id, "cancelled")
                return
            code = p.poll()
            self._set_status(task_id, "failed", error=f"推理进程异常退出(code={code})，last={last_line[:200]}")
        except Exception as e:
            self._set_status(task_id, "failed", error=str(e))


def _extract_pct(text: str) -> float | None:
    import re

    m = re.search(r"(\d{1,3}(?:\.\d+)?)%", text)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except Exception:
        return None
    if v < 0 or v > 100:
        return None
    return v


def _resolve_upload_path(upload_id: str) -> Path:
    # We store uploads as: uploads/<uploadId>_<originalName>
    for p in UPLOADS_DIR.glob(f"{upload_id}_*"):
        return p
    return UPLOADS_DIR / f"{upload_id}_missing"


def _resolve_index_path(index_id: str) -> Path | None:
    """
    Resolve index_id (relative path to ROOT_DIR, or filename) into a safe path.
    Accepts paths like 'logs/222/xxx.index' or 'assets/indices/xxx.index'.
    Returns None if not found / invalid.
    """
    raw = (index_id or "").strip()
    if not raw:
        return None

    # Remove any leading slashes to avoid treating it as absolute
    raw = raw.lstrip("/\\")

    # Try as path relative to ROOT_DIR first (new format from _list_indexes)
    p_root = ROOT_DIR / raw
    try:
        p_root_resolved = p_root.resolve()
        # Must be under ROOT_DIR for safety
        _ = p_root_resolved.relative_to(ROOT_DIR)
        if p_root_resolved.exists() and p_root_resolved.is_file():
            return p_root_resolved
    except Exception:
        pass

    # Fallback: try under INDICES_DIR (legacy format)
    candidates = [raw, os.path.basename(raw)]
    for cand in candidates:
        p0 = INDICES_DIR / cand
        try:
            p0_resolved = p0.resolve()
        except Exception:
            p0_resolved = p0

        try:
            _ = p0_resolved.relative_to(INDICES_DIR)
        except Exception:
            pass
        else:
            if p0_resolved.exists() and p0_resolved.is_file():
                return p0_resolved

        # Handle symlinks
        try:
            if os.path.islink(p0):
                link_target = os.readlink(p0)
            else:
                link_target = None
        except Exception:
            link_target = None

        if link_target:
            for base in (p0.parent, ROOT_DIR):
                try:
                    pt = (Path(base) / link_target).resolve()
                except Exception:
                    continue
                try:
                    _ = pt.relative_to(ROOT_DIR)
                except Exception:
                    continue
                if pt.exists() and pt.is_file():
                    return pt
    return None


def _list_models() -> list[ModelItem]:
    if not WEIGHTS_DIR.exists():
        return []
    items: list[ModelItem] = []
    for p in sorted(WEIGHTS_DIR.glob("*.pth")):
        items.append(ModelItem(id=p.name, name=p.name))
    return items


def _list_indexes() -> list[IndexItem]:
    """
    List all .index files from both INDICES_DIR (assets/indices) and LOGS_DIR (logs).
    This matches the old WebUI behavior which scans both index_root and outside_index_root.
    Excludes files with 'trained' in the name (intermediate files).
    """
    items: list[IndexItem] = []
    seen_ids: set[str] = set()

    # Scan LOGS_DIR first (primary index location, like old index_root=logs)
    if LOGS_DIR.exists():
        for p in sorted(LOGS_DIR.rglob("*.index")):
            if "trained" in p.name:
                continue
            # Use path relative to ROOT_DIR for unique identification
            rel = p.relative_to(ROOT_DIR).as_posix()
            if rel not in seen_ids:
                seen_ids.add(rel)
                items.append(IndexItem(id=rel, name=rel))

    # Scan INDICES_DIR (secondary, like old outside_index_root)
    if INDICES_DIR.exists():
        for p in sorted(INDICES_DIR.rglob("*.index")):
            if "trained" in p.name:
                continue
            rel = p.relative_to(ROOT_DIR).as_posix()
            if rel not in seen_ids:
                seen_ids.add(rel)
                items.append(IndexItem(id=rel, name=rel))

    return items


_ensure_dirs()
task_manager = TaskManager()

app = FastAPI(title="RVCv2 Web Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _on_startup() -> None:
    task_manager.start()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/infer/options", response_model=InferOptionsResp)
def infer_options() -> InferOptionsResp:
    """
    v1: Provide UI-friendly options + recommended defaults.
    Frontend should not hardcode enums/values.
    """
    f0_methods: list[str] = ["pm", "harvest", "rmvpe"]
    # crepe is optional; include only if installed/importable
    try:
        import torchcrepe  # type: ignore  # noqa: F401

        f0_methods.insert(2, "crepe")  # pm, harvest, crepe, rmvpe
    except Exception:
        pass

    resample_srs = [0, 16000, 22050, 32000, 40000, 44100, 48000]
    return InferOptionsResp(f0_methods=f0_methods, resample_srs=resample_srs, defaults=InferDefaults())


@app.get("/api/debug/paths")
def debug_paths() -> dict[str, Any]:
    # Dev-only diagnostic endpoint
    try:
        idx_files = [p.name for p in list(INDICES_DIR.rglob("*.index"))[:20]]
    except Exception:
        idx_files = []
    try:
        wt_files = [p.name for p in list(WEIGHTS_DIR.glob("*.pth"))[:20]]
    except Exception:
        wt_files = []
    return {
        "root": str(ROOT_DIR),
        "weightsDir": str(WEIGHTS_DIR),
        "indexesDir": str(INDICES_DIR),
        "storageDir": str(STORAGE_DIR),
        "weightsSample": wt_files,
        "indexesSample": idx_files,
    }


@app.get("/api/debug/resolve-index")
def debug_resolve_index(indexId: str) -> dict[str, Any]:
    p = _resolve_index_path(indexId)
    cand1 = (indexId or "").strip().lstrip("/\\")
    cand2 = os.path.basename(cand1)
    try:
        p1 = (INDICES_DIR / cand1).resolve()
        p1_exists = p1.exists()
        p1_is_file = p1.is_file()
    except Exception as e:
        p1 = None
        p1_exists = False
        p1_is_file = False
        err1 = str(e)
    else:
        err1 = None
    try:
        p2 = (INDICES_DIR / cand2).resolve()
        p2_exists = p2.exists()
        p2_is_file = p2.is_file()
    except Exception as e:
        p2 = None
        p2_exists = False
        p2_is_file = False
        err2 = str(e)
    else:
        err2 = None
    return {
        "indexId": indexId,
        "resolved": str(p) if p else None,
        "cand1": cand1,
        "path1": str(p1) if p1 else None,
        "path1_exists": p1_exists,
        "path1_is_file": p1_is_file,
        "path1_err": err1,
        "cand2": cand2,
        "path2": str(p2) if p2 else None,
        "path2_exists": p2_exists,
        "path2_is_file": p2_is_file,
        "path2_err": err2,
    }


@app.get("/api/models", response_model=list[ModelItem])
def list_models() -> list[ModelItem]:
    return _list_models()


@app.get("/api/indexes", response_model=list[IndexItem])
def list_indexes() -> list[IndexItem]:
    return _list_indexes()


@app.post("/api/uploads/audio", response_model=UploadResp)
def upload_audio(file: UploadFile = File(...)) -> UploadResp:
    upload_id = _uuid()
    filename = _safe_filename(file.filename or "audio")
    out_path = UPLOADS_DIR / f"{upload_id}_{filename}"
    with out_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return UploadResp(uploadId=upload_id, filename=filename)


@app.post("/api/uploads/f0", response_model=UploadResp)
def upload_f0(file: UploadFile = File(...)) -> UploadResp:
    upload_id = _uuid()
    filename = _safe_filename(file.filename or "f0.txt")
    out_path = UPLOADS_DIR / f"{upload_id}_{filename}"
    with out_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return UploadResp(uploadId=upload_id, filename=filename)


@app.post("/api/tasks/infer-offline", response_model=CreateTaskResp)
def create_infer_offline(req: CreateInferOfflineReq) -> CreateTaskResp:
    # Validate model exists
    model_names = {m.id for m in _list_models()}
    if req.modelId not in model_names:
        raise HTTPException(status_code=400, detail="modelId not found")
    if req.indexId:
        idx = _resolve_index_path(req.indexId)
        if idx is None:
            raise HTTPException(status_code=400, detail=f"indexId not found: {req.indexId!r}")
    task_id = task_manager.create_infer_task(req)
    return CreateTaskResp(taskId=task_id)


@app.get("/api/tasks/{task_id}", response_model=TaskInfo)
def get_task(task_id: str) -> TaskInfo:
    try:
        return task_manager.get(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="task not found")


@app.post("/api/tasks/{task_id}/cancel", response_model=TaskInfo)
def cancel_task(task_id: str) -> TaskInfo:
    try:
        return task_manager.cancel(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="task not found")


@app.get("/api/tasks/{task_id}/logs", response_model=LogsResp)
def get_task_logs(task_id: str, cursor: str | None = None, limit: int = 200) -> LogsResp:
    try:
        return task_manager.read_logs(task_id, cursor=cursor, limit=limit)
    except KeyError:
        raise HTTPException(status_code=404, detail="task not found")


def _get_media_type(path: Path) -> str:
    """Get appropriate media type for audio files to enable browser seeking."""
    ext = path.suffix.lower()
    media_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
    }
    return media_types.get(ext, "application/octet-stream")


@app.get("/api/artifacts/{artifact_id}")
def download_artifact(artifact_id: str, request: Request):
    # v1: artifacts are files named <artifactId>.<ext>
    matches = list(ARTIFACTS_DIR.glob(f"{artifact_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="artifact not found")
    path = matches[0]
    media_type = _get_media_type(path)
    
    # FileResponse supports HTTP Range requests for seeking in audio/video
    # Setting correct media_type is critical for browser to use Range requests
    return FileResponse(
        path=str(path),
        filename=path.name,
        media_type=media_type,
        headers={"Accept-Ranges": "bytes"},
    )

