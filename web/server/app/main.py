import os
import sys
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
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


@dataclass
class TaskRecord:
    info: TaskInfo
    req: CreateInferOfflineReq
    log_path: Path
    cancel: bool = False
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
            self._execute_infer_offline(task_id, rec)

    def _execute_infer_offline(self, task_id: str, rec: TaskRecord) -> None:
        self._set_status(task_id, "running", progress=0.0)
        req = rec.req
        try:
            audio_path = _resolve_upload_path(req.audioUploadId)
            if not audio_path.exists():
                raise RuntimeError("audioUploadId not found")

            f0_path: str | None = None
            if req.f0UploadId:
                p = _resolve_upload_path(req.f0UploadId)
                if p.exists():
                    f0_path = str(p)

            # Load / switch model (serial, so safe to reuse VC instance)
            if self._loaded_model_id != req.modelId:
                self._append_log(task_id, {"ts": time.time(), "kind": "info", "text": f"Loading model: {req.modelId}"})
                self._vc.get_vc(req.modelId)
                self._loaded_model_id = req.modelId

            # Resolve index path from registry (optional)
            index_path = ""
            if req.indexId:
                idx = _resolve_index_path(req.indexId)
                if idx and idx.exists():
                    index_path = str(idx)

            # Run generator and persist progress lines
            gen = self._vc.vc_single(
                0,
                str(audio_path),
                req.f0_up_key,
                f0_path,
                req.f0_method,
                "",
                index_path,
                req.index_rate,
                req.filter_radius,
                req.resample_sr,
                req.rms_mix_rate,
                req.protect,
            )

            final_audio_path: str | None = None
            last_text: str = ""
            for text, audio_out in gen:
                if rec.cancel:
                    self._set_status(task_id, "cancelled")
                    return
                last_text = str(text)
                self._append_log(task_id, {"ts": time.time(), "kind": "log", "text": last_text})
                # heuristic progress: try parse "xx.x%" from text
                pct = _extract_pct(last_text)
                if pct is not None:
                    rec.info.progress = pct
                if isinstance(audio_out, str) and audio_out:
                    final_audio_path = audio_out
            if not final_audio_path:
                raise RuntimeError("Inference finished without output audio")

            artifact_id = _uuid()
            out_ext = Path(final_audio_path).suffix or ".wav"
            dst = ARTIFACTS_DIR / f"{artifact_id}{out_ext}"
            shutil.copy2(final_audio_path, dst)
            self._set_status(task_id, "succeeded", progress=100.0, resultArtifactId=artifact_id)
            self._append_log(task_id, {"ts": time.time(), "kind": "artifact", "artifactId": artifact_id, "path": str(dst)})
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
    # v1: index_id is the filename in INDICES_DIR.
    name = os.path.basename(index_id)
    p = INDICES_DIR / name
    return p


def _list_models() -> list[ModelItem]:
    if not WEIGHTS_DIR.exists():
        return []
    items: list[ModelItem] = []
    for p in sorted(WEIGHTS_DIR.glob("*.pth")):
        items.append(ModelItem(id=p.name, name=p.name))
    return items


def _list_indexes() -> list[IndexItem]:
    if not INDICES_DIR.exists():
        return []
    items: list[IndexItem] = []
    for p in sorted(INDICES_DIR.rglob("*.index")):
        # Keep relative name under indices dir to avoid collisions
        rel = p.relative_to(INDICES_DIR).as_posix()
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
        idx = INDICES_DIR / req.indexId
        if not idx.exists():
            raise HTTPException(status_code=400, detail="indexId not found")
    task_id = task_manager.create_infer_task(req)
    return CreateTaskResp(taskId=task_id)


@app.get("/api/tasks/{task_id}", response_model=TaskInfo)
def get_task(task_id: str) -> TaskInfo:
    try:
        return task_manager.get(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="task not found")


@app.get("/api/tasks/{task_id}/logs", response_model=LogsResp)
def get_task_logs(task_id: str, cursor: str | None = None, limit: int = 200) -> LogsResp:
    try:
        return task_manager.read_logs(task_id, cursor=cursor, limit=limit)
    except KeyError:
        raise HTTPException(status_code=404, detail="task not found")


@app.get("/api/artifacts/{artifact_id}")
def download_artifact(artifact_id: str):
    # v1: artifacts are files named <artifactId>.<ext>
    matches = list(ARTIFACTS_DIR.glob(f"{artifact_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="artifact not found")
    path = matches[0]
    return FileResponse(path=str(path), filename=path.name, media_type="application/octet-stream")

