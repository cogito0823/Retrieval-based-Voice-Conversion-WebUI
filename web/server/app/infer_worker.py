# CRITICAL: Set OpenMP to single-thread mode on macOS BEFORE any imports
# to avoid OpenMP deadlock when running in a subprocess with MPS.
# Environment variables must be set before any library that uses OpenMP is loaded.
import os
import sys
if sys.platform == "darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import time
import uuid
from pathlib import Path

# Ensure proper multiprocessing behavior on macOS
# This must be done before importing torch
import multiprocessing
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

# Also set faiss single-thread via API (belt and suspenders approach)
if sys.platform == "darwin":
    try:
        import faiss
        faiss.omp_set_num_threads(1)
    except Exception:
        pass


def _uuid() -> str:
    return uuid.uuid4().hex


def _print(obj: dict):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--index-path", default="")
    parser.add_argument("--f0-file", default="")
    parser.add_argument("--f0-up-key", type=int, default=0)
    parser.add_argument("--f0-method", default="rmvpe")
    parser.add_argument("--index-rate", type=float, default=0.0)
    parser.add_argument("--filter-radius", type=int, default=3)
    parser.add_argument("--resample-sr", type=int, default=0)
    parser.add_argument("--rms-mix-rate", type=float, default=0.25)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--artifacts-dir", required=True)
    args = parser.parse_args()

    repo_root = Path(os.getcwd()).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Import torch early and ensure MPS is properly initialized on macOS
    import torch
    if sys.platform == "darwin" and torch.backends.mps.is_available():
        # Warm up MPS to avoid initialization issues in subprocess
        try:
            _ = torch.zeros(1, device="mps")
            if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        except Exception:
            pass

    # Import inside worker
    from configs.config import Config
    from infer.modules.vc.modules import VC

    # sanitize argv for Config()
    _argv = sys.argv[:]
    sys.argv = [_argv[0]]
    try:
        config = Config()
    finally:
        sys.argv = _argv
    vc = VC(config)

    # Load model
    _print(
        {
            "kind": "log",
            "text": f"参数：f0_method={args.f0_method}, index_rate={args.index_rate}, filter_radius={args.filter_radius}, resample_sr={args.resample_sr}, rms_mix_rate={args.rms_mix_rate}, protect={args.protect}, index={'yes' if args.index_path else 'no'}",
        }
    )
    _print({"kind": "progress", "pct": 1.0, "text": f"加载模型：{args.model_id}"})
    vc.get_vc(args.model_id)

    gen = vc.vc_single(
        0,
        args.audio_path,
        args.f0_up_key,
        args.f0_file or None,
        args.f0_method,
        "",
        args.index_path or "",
        args.index_rate,
        args.filter_radius,
        args.resample_sr,
        args.rms_mix_rate,
        args.protect,
    )

    final_audio_path = None
    last_text = ""
    for text, audio_out in gen:
        last_text = str(text)
        # emit as progress message (text already includes pct/eta)
        pct = None
        try:
            import re

            m = re.search(r"(\d{1,3}(?:\.\d+)?)%", last_text)
            if m:
                pct = float(m.group(1))
        except Exception:
            pct = None
        _print({"kind": "progress", "pct": pct, "text": last_text})
        if isinstance(audio_out, str) and audio_out:
            final_audio_path = audio_out

    if not final_audio_path:
        _print({"kind": "done", "success": False, "error": "未生成输出音频"})
        return 2

    if "Success." not in last_text:
        _print({"kind": "done", "success": False, "error": last_text or "推理失败"})
        return 3

    # Copy to artifacts
    art_dir = Path(args.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = _uuid()
    out_ext = Path(final_audio_path).suffix or ".wav"
    dst = art_dir / f"{artifact_id}{out_ext}"
    try:
        import shutil

        shutil.copy2(final_audio_path, dst)
    except Exception as e:
        _print({"kind": "done", "success": False, "error": f"保存产物失败：{e}"})
        return 4

    _print({"kind": "done", "success": True, "artifactId": artifact_id})
    return 0


if __name__ == "__main__":
    t0 = time.time()
    try:
        code = main()
    except Exception as e:
        _print({"kind": "done", "success": False, "error": f"worker异常：{e}"})
        code = 1
    # best-effort: flush time
    _print({"kind": "log", "text": f"worker结束，耗时 {time.time() - t0:.2f}s"})
    raise SystemExit(code)

