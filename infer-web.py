import os
import sys

# 尽量减少 OpenMP/BLAS 线程数，避免 macOS 下部分二进制库组合时触发崩溃
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from dotenv import load_dotenv

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()
from infer.modules.vc.modules import VC
from infer.modules.uvr5.modules import uvr
from infer.lib.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
from i18n.i18n import I18nAuto
from configs.config import Config
import torch, platform
import numpy as np
import gradio as gr
import fairseq
import pathlib
import json
import time
from time import sleep
from subprocess import Popen
import subprocess
from random import shuffle
import warnings
import traceback
import threading
import shutil
import logging
import faulthandler
import signal
import re
import datetime
from collections import deque
import ast


logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# 兼容性补丁：
# gradio==3.34.0 在请求被中断/退出（例如 Ctrl-C）时，
# Queue.process_events 可能触发：AttributeError: 'AsyncRequest' object has no attribute '_json_response_data'
# 这属于 gradio 内部实现细节（与业务无关），这里兜底避免退出时刷一堆 asyncio 报错。
try:
    import gradio.utils as _gr_utils

    if hasattr(_gr_utils, "AsyncRequest") and isinstance(
        getattr(_gr_utils.AsyncRequest, "json", None), property
    ):

        def _safe_asyncrequest_json(self):
            return getattr(self, "_json_response_data", {}) or {}

        _gr_utils.AsyncRequest.json = property(_safe_asyncrequest_json)
except Exception:
    pass

# 便于定位 Segmentation fault：将崩溃时的 Python 栈写入日志文件
try:
    _fh_path = os.path.join(os.getcwd(), "logs", "faulthandler.log")
    os.makedirs(os.path.dirname(_fh_path), exist_ok=True)
    _fh_file = open(_fh_path, "a", encoding="utf-8")
    faulthandler.enable(file=_fh_file, all_threads=True)
    try:
        faulthandler.register(signal.SIGSEGV, file=_fh_file, all_threads=True, chain=True)
    except Exception:
        # 某些平台不允许注册 SIGSEGV，忽略即可
        pass
except Exception:
    pass

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

# Training process control (best-effort, single active training assumed)
_TRAIN_PROC: dict[str, object] = {
    "p": None,  # subprocess.Popen
    "exp": None,
    "paused": False,
}


def _set_train_proc(p, exp_name: str):
    _TRAIN_PROC["p"] = p
    _TRAIN_PROC["exp"] = exp_name
    _TRAIN_PROC["paused"] = False


def _clear_train_proc():
    _TRAIN_PROC["p"] = None
    _TRAIN_PROC["exp"] = None
    _TRAIN_PROC["paused"] = False


def _signal_train_proc(sig: int) -> tuple[bool, str]:
    """
    Send signal to current training process (and its group on POSIX).
    """
    p = _TRAIN_PROC.get("p")
    if p is None:
        return False, "当前没有正在运行的训练进程。"
    try:
        rc = p.poll()
        if rc is not None:
            _clear_train_proc()
            return False, f"训练进程已结束（exit code={rc}）。"
    except Exception:
        pass

    try:
        if platform.system() != "Windows":
            # If started as a new session, pid is the process group leader
            os.killpg(p.pid, sig)
        else:
            # Windows: best-effort
            os.kill(p.pid, sig)
        return True, "OK"
    except Exception as e:
        return False, f"发送信号失败：{e}"


def pause_training_ui() -> str:
    if _TRAIN_PROC.get("paused"):
        return "当前已暂停。"
    ok, msg = _signal_train_proc(signal.SIGSTOP if hasattr(signal, "SIGSTOP") else signal.SIGTERM)
    if ok and hasattr(signal, "SIGSTOP"):
        _TRAIN_PROC["paused"] = True
        return "已暂停训练（SIGSTOP）。"
    return f"暂停失败：{msg}"


def resume_training_ui() -> str:
    if not _TRAIN_PROC.get("paused"):
        return "当前未处于暂停状态。"
    ok, msg = _signal_train_proc(signal.SIGCONT if hasattr(signal, "SIGCONT") else signal.SIGTERM)
    if ok and hasattr(signal, "SIGCONT"):
        _TRAIN_PROC["paused"] = False
        return "已继续训练（SIGCONT）。"
    return f"继续失败：{msg}"


def stop_training_ui() -> str:
    # Graceful stop: TERM -> KILL fallback
    if _TRAIN_PROC.get("p") is None:
        return "当前没有正在运行的训练进程。"
    ok, msg = _signal_train_proc(signal.SIGTERM)
    if ok:
        return "已发送停止信号（SIGTERM）。"
    return f"停止失败：{msg}"


def get_train_runtime_status_ui(exp_name: str = "") -> str:
    """
    Show whether training is running (WebUI-launched or any system process).
    """
    exp_name = (exp_name or "").strip()
    lines: list[str] = []

    # WebUI-launched process
    p = _TRAIN_PROC.get("p")
    if p is not None:
        try:
            rc = p.poll()
            if rc is None:
                exists, cmd = _probe_pid(int(p.pid))
                ok_train = exists and _is_train_cmd(cmd, str(_TRAIN_PROC.get("exp") or "") or None)
                lines.append(
                    f"WebUI 训练进程：运行中 pid={p.pid} exp={_TRAIN_PROC.get('exp')} paused={_TRAIN_PROC.get('paused')}"
                )
                if exists:
                    lines.append(f"  cmd: {cmd}")
                else:
                    lines.append("  注意：pid 不存在（可能已退出）。")
                if exists and not ok_train:
                    lines.append("  注意：该 pid 不是训练进程（cmd 不匹配）。")
            else:
                lines.append(f"WebUI 训练进程：已结束 exit code={rc}")
        except Exception as e:
            lines.append(f"WebUI 训练进程：状态获取失败：{e}")
    else:
        lines.append("WebUI 训练进程：无")

    # Latest train_start pid from events (per exp)
    if exp_name:
        try:
            events_path = os.path.join(now_dir, "logs", exp_name, "train_events.jsonl")
            evs = _tail_jsonl_file(events_path, max_chars=200000)
            last_start = None
            for ev in reversed(evs):
                if isinstance(ev, dict) and ev.get("type") == "train_start":
                    last_start = ev
                    break
            if last_start and last_start.get("pid") is not None:
                pid = int(last_start["pid"])
                exists, cmd = _probe_pid(pid)
                if exists and _is_train_cmd(cmd, exp_name):
                    lines.append(f"最近一次训练 pid={pid}：运行中（命令匹配）")
                elif exists:
                    lines.append(f"最近一次训练 pid={pid}：pid 存在但不是训练进程（命令不匹配）")
                    lines.append(f"  cmd: {cmd}")
                else:
                    lines.append(f"最近一次训练 pid={pid}：未运行（进程不存在/已结束）")
            else:
                lines.append("最近一次训练：未发现 train_start 事件。")
        except Exception as e:
            lines.append(f"最近一次训练：读取事件失败：{e}")

    # System scan (best-effort)
    try:
        out = subprocess.check_output(["ps", "aux"], text=True)
        proc_lines = [l for l in out.splitlines() if "infer/modules/train/train.py" in l]
        if exp_name:
            proc_lines = [l for l in proc_lines if f" -e {exp_name} " in l or l.endswith(f" -e {exp_name}")]
        if proc_lines:
            lines.append("系统检测到训练进程：")
            lines.extend(proc_lines[:5])
            if len(proc_lines) > 5:
                lines.append(f"...（共 {len(proc_lines)} 条，已截断）")
        else:
            lines.append("系统检测：未发现训练进程。")
    except Exception as e:
        lines.append(f"系统检测失败：{e}")

    return "\n".join(lines)


config = Config()
vc = VC(config)


if config.dml == True:

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
i18n = I18nAuto()
logger.info(i18n)
# 判断是否有能用来训练和加速推理的N卡
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
                "10",
                "16",
                "20",
                "30",
                "40",
                "A2",
                "A3",
                "A4",
                "P4",
                "A50",
                "500",
                "A60",
                "70",
                "80",
                "90",
                "M4",
                "T4",
                "TITAN",
                "4060",
                "L",
                "6000",
            ]
        ):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")
outside_index_root = os.getenv("outside_index_root")

names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []


def lookup_indices(index_root):
    global index_paths
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))


lookup_indices(index_root)
lookup_indices(outside_index_root)
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))


def change_choices():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }


def clean():
    return {"value": "", "__type__": "update"}


def export_onnx(ModelPath, ExportedPath):
    from infer.modules.onnx.export import export_onnx as eo

    eo(ModelPath, ExportedPath)


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        # poll==None代表进程未结束
        # 只要有一个进程未结束都不停
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


def tail_text_file(path: str, max_chars: int = 12000) -> str:
    """
    Read the tail of a text file (best-effort).
    Keeps UI responsive even when logs get large.
    """
    try:
        if not os.path.exists(path):
            return ""
        # Read from end to avoid loading huge files
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - max_chars)
            f.seek(start, os.SEEK_SET)
            data = f.read()
        # If we started mid-line, drop the first partial line
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = data.decode(errors="replace")
        if start > 0 and "\n" in text:
            text = text.split("\n", 1)[1]
        return text
    except Exception:
        return traceback.format_exc()


def _safe_list_log_experiments() -> list[str]:
    """
    List experiment folders under ./logs (best-effort).
    """
    try:
        base = os.path.join(now_dir, "logs")
        if not os.path.isdir(base):
            return []
        exps = []
        for name in os.listdir(base):
            if name.startswith("."):
                continue
            p = os.path.join(base, name)
            if os.path.isdir(p):
                exps.append(name)
        exps.sort()
        return exps
    except Exception:
        return []


def _exp_meta_path(exp_name: str) -> str:
    return os.path.join(now_dir, "logs", exp_name, "meta.json")


def _write_exp_meta(exp_name: str, trainset_dir: str | None = None):
    try:
        if not exp_name:
            return
        p = _exp_meta_path(exp_name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        meta = {}
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    meta = json.load(f) or {}
            except Exception:
                meta = {}
        meta.setdefault("created_at", time.time())
        meta["updated_at"] = time.time()
        if trainset_dir:
            meta["trainset_dir"] = trainset_dir
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except Exception:
        pass


def _guess_trainset_dir_from_preprocess_log(exp_name: str) -> str:
    """
    Best-effort heuristic:
    - Read logs/<exp>/preprocess.log lines like "<file>\t-> Success"
    - Compute common path prefix; return directory.
    """
    try:
        p = os.path.join(now_dir, "logs", exp_name, "preprocess.log")
        if not os.path.exists(p):
            return ""
        files = []
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if "\t->" in line:
                    fp = line.split("\t->", 1)[0].strip()
                    if fp:
                        files.append(fp)
        if not files:
            return ""
        common = os.path.commonpath(files)
        if os.path.isfile(common):
            common = os.path.dirname(common)
        if os.path.isdir(common):
            return common
        # fallback: dirname of first file
        return os.path.dirname(files[0])
    except Exception:
        return ""


def _load_exp_settings(exp_name: str) -> tuple[str, str, str, bool, str]:
    """
    Returns: (exp_name, trainset_dir, sr2, if_f0, version)
    """
    exp_name = (exp_name or "").strip()
    if not exp_name:
        return "", "", "40k", True, "v2"

    # defaults
    trainset_dir = ""
    sr2 = "40k"
    if_f0 = True
    version = "v2"

    # 1) meta.json
    try:
        mp = _exp_meta_path(exp_name)
        if os.path.exists(mp):
            with open(mp, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
            trainset_dir = str(meta.get("trainset_dir", "")).strip()
    except Exception:
        pass

    # 2) train.log first line (contains hps dict, usually includes sample_rate/version/if_f0)
    try:
        tl = os.path.join(now_dir, "logs", exp_name, "train.log")
        if os.path.exists(tl):
            with open(tl, "r", encoding="utf-8", errors="replace") as f:
                first = f.readline().strip()
            if "\tINFO\t" in first:
                payload = first.split("\tINFO\t", 1)[1].strip()
                data = ast.literal_eval(payload)
                if isinstance(data, dict):
                    sr2 = str(data.get("sample_rate", sr2))
                    version = str(data.get("version", version))
                    if_f0 = bool(int(data.get("if_f0", 1)))
    except Exception:
        pass

    # 3) infer version from feature folder if missing
    try:
        exp_dir = os.path.join(now_dir, "logs", exp_name)
        if os.path.isdir(os.path.join(exp_dir, "3_feature256")):
            version = "v1"
        elif os.path.isdir(os.path.join(exp_dir, "3_feature768")):
            version = "v2"
        if os.path.isdir(os.path.join(exp_dir, "2a_f0")) and os.path.isdir(
            os.path.join(exp_dir, "2b-f0nsf")
        ):
            if_f0 = True
    except Exception:
        pass

    # 4) preprocess.log heuristic if trainset_dir still unknown
    if not trainset_dir:
        trainset_dir = _guess_trainset_dir_from_preprocess_log(exp_name)

    # sanitize
    if sr2 not in ("32k", "40k", "48k"):
        # map by sampling rate numbers if any
        if "48000" in sr2:
            sr2 = "48k"
        elif "40000" in sr2:
            sr2 = "40k"
        elif "32000" in sr2:
            sr2 = "32k"
        else:
            sr2 = "40k"

    if version not in ("v1", "v2"):
        version = "v2"

    if version == "v1" and sr2 == "32k":
        sr2 = "40k"

    return exp_name, trainset_dir, sr2, if_f0, version


def _apply_exp_selection(exp_name: str):
    exp_name, trainset_dir, sr2_val, if_f0_val, version_val = _load_exp_settings(exp_name)
    # Update pretrained defaults to match selection
    path_str = "" if version_val == "v1" else "_v2"
    f0_str = "f0" if if_f0_val else ""
    pg, pd = get_pretrained_models(path_str, f0_str, sr2_val)

    exp_dir = os.path.join(now_dir, "logs", exp_name)

    # Preprocess / feature extraction snapshots
    preprocess_log = os.path.join(exp_dir, "preprocess.log")
    extract_log = os.path.join(exp_dir, "extract_f0_feature.log")
    preprocess_tail = tail_text_file(preprocess_log, max_chars=4000) if os.path.exists(preprocess_log) else ""
    extract_tail = tail_text_file(extract_log, max_chars=6000) if os.path.exists(extract_log) else ""

    # Summaries based on artifact folders (cheap)
    def _count_children(p: str) -> int:
        try:
            if not os.path.isdir(p):
                return 0
            return len([x for x in os.listdir(p) if not x.startswith(".")])
        except Exception:
            return 0

    gt_cnt = _count_children(os.path.join(exp_dir, "0_gt_wavs"))
    wav16_cnt = _count_children(os.path.join(exp_dir, "1_16k_wavs"))
    f0_cnt = _count_children(os.path.join(exp_dir, "2a_f0"))
    f0nsf_cnt = _count_children(os.path.join(exp_dir, "2b-f0nsf"))
    fea_dir = "3_feature256" if version_val == "v1" else "3_feature768"
    fea_cnt = _count_children(os.path.join(exp_dir, fea_dir))

    preprocess_info = (
        f"[{exp_name}] preprocess: 0_gt_wavs={gt_cnt}, 1_16k_wavs={wav16_cnt}\n"
        + (preprocess_tail.strip() or "（无 preprocess.log）")
    )
    extract_info = (
        f"[{exp_name}] feature: {fea_dir}={fea_cnt}, 2a_f0={f0_cnt}, 2b-f0nsf={f0nsf_cnt}\n"
        + (extract_tail.strip() or "（无 extract_f0_feature.log）")
    )

    # Training snapshot from events/log tail
    train_log_path = os.path.join(exp_dir, "train.log")
    train_log_tail = tail_text_file(train_log_path, max_chars=12000) if os.path.exists(train_log_path) else ""
    events_path = os.path.join(exp_dir, "train_events.jsonl")

    # Load hparams snapshot from the latest train_start event (preferred),
    # or fallback to the latest INFO-hps line in train.log (NOT the first line).
    save_every_epoch = None
    total_epoch = None
    batch_size = None
    log_interval = None
    if_latest = None
    if_cache = None
    save_every_weights = None
    gpus = None
    pretrainG = None
    pretrainD = None
    # 0) latest train_start event from JSONL
    try:
        evs = _tail_jsonl_file(events_path)
        for ev in reversed(evs):
            if isinstance(ev, dict) and ev.get("type") == "train_start":
                save_every_epoch = ev.get("save_every_epoch")
                total_epoch = ev.get("total_epoch")
                batch_size = ev.get("batch_size")
                log_interval = ev.get("log_interval")
                if_latest = ev.get("if_latest")
                if_cache = ev.get("if_cache_data_in_gpu")
                save_every_weights = ev.get("save_every_weights")
                gpus = ev.get("gpus")
                pretrainG = ev.get("pretrainG")
                pretrainD = ev.get("pretrainD")
                break
    except Exception:
        pass
    # 1) fallback: latest INFO-hps dict line from train.log
    if total_epoch is None and os.path.exists(train_log_path):
        try:
            with open(train_log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
            last_payload = None
            for line in reversed(lines[-500:]):
                if "\tINFO\t{" in line:
                    last_payload = line.split("\tINFO\t", 1)[1].strip()
                    break
            if last_payload:
                data = ast.literal_eval(last_payload)
                if isinstance(data, dict):
                    save_every_epoch = data.get("save_every_epoch")
                    total_epoch = data.get("total_epoch")
                    gpus = data.get("gpus")
                    if_latest = data.get("if_latest")
                    if_cache = data.get("if_cache_data_in_gpu")
                    save_every_weights = data.get("save_every_weights")
                    pretrainG = data.get("pretrainG")
                    pretrainD = data.get("pretrainD")
                    tr = data.get("train") or {}
                    if isinstance(tr, dict):
                        batch_size = tr.get("batch_size")
                        log_interval = tr.get("log_interval")
        except Exception:
            pass

    def _u_int(v, default=None):
        try:
            if v is None:
                return default
            return int(v)
        except Exception:
            return default

    save_every_epoch_u = _u_int(save_every_epoch)
    total_epoch_u = _u_int(total_epoch)
    batch_size_u = _u_int(batch_size)
    log_interval_u = _u_int(log_interval)
    if_latest_u = _u_int(if_latest)
    if_cache_u = _u_int(if_cache)
    save_every_weights_u = _u_int(save_every_weights)

    # Build UI updates for training controls (only update when we have values)
    save_epoch_update = {"__type__": "update"}
    total_epoch_update = {"__type__": "update"}
    batch_size_update = {"__type__": "update"}
    log_interval_update = {"__type__": "update"}
    if_latest_update = {"__type__": "update"}
    if_cache_update = {"__type__": "update"}
    save_weights_update = {"__type__": "update"}
    gpus_update = {"__type__": "update"}
    pg_update = pg
    pd_update = pd

    if save_every_epoch_u is not None:
        save_epoch_update = {"__type__": "update", "value": save_every_epoch_u}
    if total_epoch_u is not None:
        total_epoch_update = {"__type__": "update", "value": total_epoch_u}
    if batch_size_u is not None:
        batch_size_update = {"__type__": "update", "value": batch_size_u}
    if log_interval_u is not None:
        log_interval_update = {"__type__": "update", "value": log_interval_u}
    if if_latest_u is not None:
        if_latest_update = {"__type__": "update", "value": i18n("是") if if_latest_u == 1 else i18n("否")}
    if if_cache_u is not None:
        if_cache_update = {"__type__": "update", "value": i18n("是") if if_cache_u == 1 else i18n("否")}
    if save_every_weights_u is not None:
        save_weights_update = {"__type__": "update", "value": i18n("是") if save_every_weights_u == 1 else i18n("否")}
    if gpus:
        gpus_update = {"__type__": "update", "value": str(gpus)}
    if pretrainG:
        pg_update = str(pretrainG)
    if pretrainD:
        pd_update = str(pretrainD)

    # Determine total_epoch from config.json if possible (for correct %)
    total_epoch_int = 0
    try:
        cfg_path = os.path.join(exp_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f) or {}
            total_epoch_int = int(cfg.get("train", {}).get("epochs", 0))  # fallback, not used for training stop
            # Prefer UI total_epoch semantics stored in hps in train.log if present
    except Exception:
        total_epoch_int = 0

    # Build monitor snapshot
    try:
        monitor = TrainMonitor(total_epoch=0, avg_window=50)
        evs = _tail_jsonl_file(events_path)
        if evs:
            monitor.update(evs)
        rendered = monitor.render(stall_threshold_sec=60)
        state_str = str(rendered.get("state") or i18n("未开始"))

        # PID-based reality check: if last train_start pid is not running, clearly mark as NOT running
        last_pid = None
        for ev in reversed(evs or []):
            if isinstance(ev, dict) and ev.get("type") == "train_start" and ev.get("pid") is not None:
                try:
                    last_pid = int(ev.get("pid"))
                except Exception:
                    last_pid = None
                break
        if last_pid is not None:
            exists, cmd = _probe_pid(last_pid)
            is_train = exists and _is_train_cmd(cmd, exp_name)
            if (not exists) or (not is_train):
                if "训练中" in state_str or "可能卡住" in state_str:
                    state_str = f"未运行（上次 pid={last_pid} 已结束/不匹配） | " + state_str
                rendered["diag"] = (
                    (str(rendered.get("diag") or "").strip() + "\n\n").strip()
                    + f"进程状态：最近一次 train_start pid={last_pid} 当前未运行（或 cmd 不匹配）。"
                ).strip()
        # If no events, fallback to legacy parser
        if not evs and train_log_tail.strip():
            progress, status = parse_train_progress(train_log_tail, 0)
            state_str = status
            rendered["epoch_pct"] = progress
            rendered["step_pct"] = 0
            rendered["elapsed_sec"] = None
            rendered["eta_sec"] = None
            rendered["eta_note"] = ""
            rendered["avg_step_sec"] = None
            rendered["last_step_sec"] = None
            rendered["losses"] = {}
        time_html, speed_html, loss_html = _render_train_cards_html(state_str, rendered)
        train_summary = str(rendered.get("summary") or "")
        train_events_txt = str(rendered.get("events") or "")
        train_diag_txt = str(rendered.get("diag") or "")
        epoch_pct = int(rendered.get("epoch_pct") or 0)
        step_pct = int(rendered.get("step_pct") or 0)
    except Exception:
        state_str = i18n("未开始")
        epoch_pct = 0
        step_pct = 0
        time_html, speed_html, loss_html = _render_train_cards_html(state_str, {"elapsed_sec": None, "eta_sec": None, "eta_note": "", "avg_step_sec": None, "last_step_sec": None, "losses": {}})
        train_diag_txt = ""
        train_events_txt = ""
        train_summary = ""

    info_msg = (
        f"已加载实验：{exp_name}\n"
        f"- 训练集路径：{trainset_dir or '（未知）'}\n"
        f"- sr：{sr2_val}\n"
        f"- version：{version_val}\n"
        f"- if_f0：{if_f0_val}\n"
        f"- preprocess: 0_gt_wavs={gt_cnt}, 1_16k_wavs={wav16_cnt}\n"
        f"- feature: {fea_dir}={fea_cnt}\n"
        f"- train: {state_str}"
    )

    return (
        exp_name,
        trainset_dir,
        {"__type__": "update", "value": sr2_val},
        {"__type__": "update", "value": bool(if_f0_val)},
        {"__type__": "update", "value": version_val},
        pg_update,
        pd_update,
        info_msg,
        preprocess_info,
        extract_info,
        save_epoch_update,
        total_epoch_update,
        batch_size_update,
        log_interval_update,
        gpus_update,
        if_latest_update,
        if_cache_update,
        save_weights_update,
        get_train_runtime_status_ui(exp_name),
        state_str,
        epoch_pct,
        step_pct,
        time_html,
        speed_html,
        loss_html,
        train_diag_txt,
        train_events_txt,
        train_summary,
        train_log_tail,
        # legacy hidden outputs
        train_summary or train_log_tail,
        epoch_pct,
        state_str,
    )


def _refresh_exp_choices():
    return {"__type__": "update", "choices": _safe_list_log_experiments()}


def _browse_directory(current_value: str):
    """
    Server-side native directory picker (tkinter). Works for local WebUI.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        initial = current_value if current_value and os.path.isdir(current_value) else now_dir
        selected = filedialog.askdirectory(initialdir=initial)
        root.destroy()
        if selected:
            return selected
        return current_value
    except Exception:
        return current_value


def _tail_jsonl_file(path: str, max_chars: int = 200000) -> list[dict]:
    """
    Read tail of a JSONL file and parse objects (best-effort).
    """
    try:
        if not os.path.exists(path):
            return []
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - max_chars)
            f.seek(start, os.SEEK_SET)
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        if start > 0 and "\n" in text:
            text = text.split("\n", 1)[1]
        out: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
        return out
    except Exception:
        return []


def _probe_pid(pid: int) -> tuple[bool, str]:
    """
    Returns (exists, cmdline). Best-effort on macOS/Linux.
    """
    try:
        out = subprocess.check_output(["ps", "-p", str(int(pid)), "-o", "command="], text=True).strip()
        if not out:
            return False, ""
        return True, out
    except Exception:
        return False, ""


def _is_train_cmd(cmdline: str, exp_name: str | None = None) -> bool:
    cmdline = cmdline or ""
    if "infer/modules/train/train.py" not in cmdline:
        return False
    if exp_name:
        # conservative match for "-e <exp>"
        needle = f" -e {exp_name} "
        if needle in cmdline:
            return True
        # sometimes command line ends with exp
        if cmdline.endswith(f" -e {exp_name}") or cmdline.endswith(f" -e {exp_name} "):
            return True
        # fallback: exp appears after -e
        if f" -e {exp_name}" in cmdline:
            return True
        return False
    return True


class JsonlTailer:
    """
    Incrementally read JSONL file (best-effort).
    Keeps an internal byte offset; safe if file does not exist yet.
    """

    def __init__(self, path: str):
        self.path = path
        self._offset = 0

    def read_new(self) -> list[dict]:
        try:
            if not os.path.exists(self.path):
                return []
            with open(self.path, "rb") as f:
                f.seek(self._offset, os.SEEK_SET)
                data = f.read()
                self._offset = f.tell()
            if not data:
                return []
            text = data.decode("utf-8", errors="replace")
            out: list[dict] = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
            return out
        except Exception:
            return []


def _format_hhmmss(seconds: float | None) -> str:
    try:
        if seconds is None:
            return "—"
        seconds = max(0, float(seconds))
        return str(datetime.timedelta(seconds=int(seconds)))
    except Exception:
        return "—"


class TrainMonitor:
    """
    Aggregates training events into stable UI metrics.
    """

    def __init__(self, total_epoch: int, avg_window: int = 50):
        self.total_epoch = max(0, int(total_epoch))
        self.avg_window = max(5, int(avg_window))
        self.start_wall = time.time()
        self.start_ts: float | None = None

        self.device = ""
        self.device_type = ""
        self.batch_size: int | None = None

        self.steps_per_epoch: int | None = None
        self.epoch: int | None = None
        self.batch_idx: int | None = None
        self.global_step: int | None = None

        self.last_step_ts: float | None = None
        self.last_step_time: float | None = None
        self.step_times = deque(maxlen=self.avg_window)

        self.losses: dict[str, float] = {}
        self.key_events = deque(maxlen=50)
        self.summary_lines = deque(maxlen=200)

    def update(self, events: list[dict]):
        for ev in events:
            et = str(ev.get("type", ""))
            ts = ev.get("ts")
            if self.start_ts is None and isinstance(ts, (int, float)):
                self.start_ts = float(ts)

            if et == "train_start":
                self.device = str(ev.get("device", ""))[:64]
                self.device_type = str(ev.get("device_type", ""))[:16]
                try:
                    self.batch_size = int(ev.get("batch_size")) if ev.get("batch_size") is not None else None
                except Exception:
                    self.batch_size = None
                self.steps_per_epoch = ev.get("steps_per_epoch") or self.steps_per_epoch
                if ev.get("total_epoch") is not None:
                    try:
                        self.total_epoch = int(ev.get("total_epoch"))
                    except Exception:
                        pass
                self.key_events.appendleft("训练启动")
            elif et == "epoch_start":
                try:
                    self.epoch = int(ev.get("epoch"))
                except Exception:
                    pass
                if self.epoch is not None:
                    self.key_events.appendleft(f"开始 Epoch {self.epoch}")
            elif et == "epoch_end":
                try:
                    self.epoch = int(ev.get("epoch"))
                except Exception:
                    pass
                if self.epoch is not None:
                    self.key_events.appendleft(f"完成 Epoch {self.epoch}")
            elif et == "checkpoint_saved":
                e = ev.get("epoch")
                gs = ev.get("global_step")
                self.key_events.appendleft(f"保存 ckpt (epoch={e}, step={gs})")
            elif et == "error":
                self.key_events.appendleft("训练异常（见日志）")
            elif et == "train_end":
                self.key_events.appendleft("训练结束")
            elif et == "step_end":
                if ev.get("epoch") is not None:
                    try:
                        self.epoch = int(ev.get("epoch"))
                    except Exception:
                        pass
                if "batch_idx" in ev and ev.get("batch_idx") is not None:
                    try:
                        self.batch_idx = int(ev.get("batch_idx"))
                    except Exception:
                        pass
                if ev.get("global_step") is not None:
                    try:
                        self.global_step = int(ev.get("global_step"))
                    except Exception:
                        pass
                if ev.get("steps_per_epoch") is not None:
                    try:
                        self.steps_per_epoch = int(ev.get("steps_per_epoch"))
                    except Exception:
                        pass

                st = ev.get("step_time_total")
                if isinstance(st, (int, float)) and st > 0:
                    self.last_step_time = float(st)
                    self.step_times.append(float(st))
                if isinstance(ts, (int, float)):
                    self.last_step_ts = float(ts)

                for k in ("loss_disc", "loss_gen", "loss_fm", "loss_mel", "loss_kl"):
                    v = ev.get(k)
                    if isinstance(v, (int, float)):
                        self.losses[k] = float(v)

                # summary line (only when losses are present to avoid spam)
                if "loss_mel" in ev:
                    spe = self.steps_per_epoch
                    s_in_epoch = (self.batch_idx + 1) if (self.batch_idx is not None) else None
                    self.summary_lines.append(
                        f"E{self.epoch}/{self.total_epoch} "
                        f"S{s_in_epoch or '—'}/{spe or '—'} "
                        f"step={self.global_step or '—'} "
                        f"t={float(ev.get('step_time_total', 0)):.3f}s "
                        f"mel={self.losses.get('loss_mel', float('nan')):.3f} "
                        f"kl={self.losses.get('loss_kl', float('nan')):.3f}"
                    )

    def render(self, stall_threshold_sec: int = 60) -> dict[str, object]:
        now = time.time()
        start = self.start_ts if self.start_ts is not None else self.start_wall
        elapsed = now - start

        epoch = int(self.epoch) if self.epoch is not None else 0
        total = int(self.total_epoch) if self.total_epoch is not None else 0
        spe = int(self.steps_per_epoch) if self.steps_per_epoch else None
        s_in_epoch = (int(self.batch_idx) + 1) if self.batch_idx is not None else None

        if total > 0 and spe and s_in_epoch:
            overall = ((epoch - 1) * spe + (s_in_epoch - 1)) / (total * spe) * 100.0
            epoch_pct = int(max(0, min(100, round(overall))))
            step_pct = int(max(0, min(100, round((s_in_epoch / spe) * 100.0))))
        elif total > 0 and epoch > 0:
            epoch_pct = int(max(0, min(100, round(epoch / total * 100.0))))
            step_pct = 0
        else:
            epoch_pct = 0
            step_pct = 0

        avg_step = None
        if self.step_times:
            avg_step = sum(self.step_times) / len(self.step_times)

        eta = None
        eta_note = ""
        if total > 0 and spe and s_in_epoch and avg_step is not None:
            remaining_steps = max(0, (total - epoch) * spe + (spe - s_in_epoch))
            eta = remaining_steps * avg_step
            eta_note = f"（基于最近 {len(self.step_times)} step 均值）"
        elif avg_step is None:
            eta_note = "（等待 step 数据…）"
        else:
            eta_note = "（等待 steps_per_epoch…）"

        stall = False
        stall_age = None
        if self.last_step_ts is not None:
            stall_age = now - self.last_step_ts
            stall = stall_age > stall_threshold_sec

        badge = "训练中"
        if stall:
            badge = "可能卡住"
        if self.key_events and self.key_events[0] == "训练结束":
            badge = "已完成"

        device_part = ""
        if self.device_type or self.device:
            device_part = f"device={self.device_type or self.device}"
        step_part = f"Step {s_in_epoch or '—'}/{spe or '—'}"

        state_line = f"{badge} | Epoch {epoch}/{total or '—'} | {step_part} | {device_part}".strip()
        time_line = f"已运行：{_format_hhmmss(elapsed)} | ETA：{_format_hhmmss(eta)} {eta_note}".strip()
        if stall and stall_age is not None:
            time_line += f" | 最后 step：{int(stall_age)}s 前"

        speed_line = "Step耗时：—"
        if avg_step is not None:
            speed_line = f"Step耗时：avg {avg_step:.3f}s（最近{len(self.step_times)}） / last {(self.last_step_time or 0):.3f}s"

        loss_parts = []
        for k in ("loss_mel", "loss_kl", "loss_gen", "loss_disc"):
            if k in self.losses:
                loss_parts.append(f"{k.replace('loss_', '')}={self.losses[k]:.3f}")
        loss_line = " | ".join(loss_parts) if loss_parts else "Loss：—"

        diag = ""
        if stall:
            diag = (
                "可能卡住：长时间未观察到 step 推进。\n"
                "- 若刚调高 workers：尝试将 DataLoader workers 改为 0/2/4 重新试\n"
                "- 若日志/写图太频繁：增大 log_interval\n"
                "- 若内存压力大：减小 batch_size 或 segment_size\n"
                "- 也可能在保存 ckpt/写盘：稍等观察 raw log"
            )

        return {
            "state": state_line,
            "epoch_pct": epoch_pct,
            "step_pct": step_pct,
            "elapsed_sec": float(elapsed),
            "eta_sec": float(eta) if eta is not None else None,
            "eta_note": eta_note,
            "avg_step_sec": float(avg_step) if avg_step is not None else None,
            "last_step_sec": float(self.last_step_time) if self.last_step_time is not None else None,
            "losses": dict(self.losses),
            "diag": diag,
            "events": "\n".join(list(self.key_events)[:12]),
            "summary": "\n".join(list(self.summary_lines)[-30:]),
        }


def _render_train_cards_html(state: str, rendered: dict[str, object]) -> tuple[str, str, str]:
    """
    Return (time_html, speed_html, loss_html)
    """
    elapsed = rendered.get("elapsed_sec")
    eta = rendered.get("eta_sec")
    eta_note = str(rendered.get("eta_note") or "")
    avg_step = rendered.get("avg_step_sec")
    last_step = rendered.get("last_step_sec")
    losses = rendered.get("losses") or {}
    if not isinstance(losses, dict):
        losses = {}

    # simple state color
    badge = "info"
    if "可能卡住" in (state or ""):
        badge = "warn"
    if "已完成" in (state or ""):
        badge = "ok"
    if "异常" in (state or ""):
        badge = "err"

    def fmt(sec):
        return _format_hhmmss(sec) if sec is not None else "—"

    time_html = f"""
<div style="display:flex; gap:12px; flex-wrap:wrap;">
  <div style="flex:1; min-width:160px; padding:12px; border:1px solid #e6e6e6; border-radius:10px;">
    <div style="font-size:12px; color:#666;">已运行</div>
    <div style="font-size:22px; font-weight:700; font-variant-numeric: tabular-nums;">{fmt(elapsed)}</div>
  </div>
  <div style="flex:1; min-width:160px; padding:12px; border:1px solid #e6e6e6; border-radius:10px;">
    <div style="font-size:12px; color:#666;">ETA</div>
    <div style="font-size:22px; font-weight:700; font-variant-numeric: tabular-nums;">{fmt(eta)}</div>
    <div style="font-size:12px; color:#888; margin-top:4px;">{eta_note}</div>
  </div>
</div>
"""

    speed_html = f"""
<div style="display:flex; gap:12px; flex-wrap:wrap;">
  <div style="flex:1; min-width:160px; padding:12px; border:1px solid #e6e6e6; border-radius:10px;">
    <div style="font-size:12px; color:#666;">平均 step</div>
    <div style="font-size:22px; font-weight:700; font-variant-numeric: tabular-nums;">{(f"{avg_step:.3f}s" if isinstance(avg_step,(int,float)) else "—")}</div>
  </div>
  <div style="flex:1; min-width:160px; padding:12px; border:1px solid #e6e6e6; border-radius:10px;">
    <div style="font-size:12px; color:#666;">最近 step</div>
    <div style="font-size:22px; font-weight:700; font-variant-numeric: tabular-nums;">{(f"{last_step:.3f}s" if isinstance(last_step,(int,float)) else "—")}</div>
  </div>
</div>
"""

    def lossv(k):
        v = losses.get(k)
        return f"{float(v):.3f}" if isinstance(v, (int, float)) else "—"

    loss_html = f"""
<div style="display:flex; gap:12px; flex-wrap:wrap;">
  <div style="flex:1; min-width:120px; padding:10px; border:1px solid #e6e6e6; border-radius:10px;">
    <div style="font-size:12px; color:#666;">mel</div>
    <div style="font-size:18px; font-weight:700; font-variant-numeric: tabular-nums;">{lossv("loss_mel")}</div>
  </div>
  <div style="flex:1; min-width:120px; padding:10px; border:1px solid #e6e6e6; border-radius:10px;">
    <div style="font-size:12px; color:#666;">kl</div>
    <div style="font-size:18px; font-weight:700; font-variant-numeric: tabular-nums;">{lossv("loss_kl")}</div>
  </div>
  <div style="flex:1; min-width:120px; padding:10px; border:1px solid #e6e6e6; border-radius:10px;">
    <div style="font-size:12px; color:#666;">gen</div>
    <div style="font-size:18px; font-weight:700; font-variant-numeric: tabular-nums;">{lossv("loss_gen")}</div>
  </div>
  <div style="flex:1; min-width:120px; padding:10px; border:1px solid #e6e6e6; border-radius:10px;">
    <div style="font-size:12px; color:#666;">disc</div>
    <div style="font-size:18px; font-weight:700; font-variant-numeric: tabular-nums;">{lossv("loss_disc")}</div>
  </div>
</div>
"""

    return time_html, speed_html, loss_html


_RE_TRAIN_EPOCH_PCT = re.compile(r"Train Epoch:\s*(\d+)\s*\[\s*(\d+)%\s*\]")
_RE_EPOCH_DONE = re.compile(r"====>\s*Epoch:\s*(\d+)\b")
_RE_EPOCH_ELAPSED = re.compile(r"====>\s*Epoch:\s*(\d+).*?\|\s*\(([^)]+)\)")


def parse_train_progress(log_text: str, total_epoch):
    """
    Parse training progress from train.log tail.
    Returns: (progress_pct: int, status: str)
    """
    def _parse_timedelta_seconds(td_str: str):
        # Expect formats like "0:05:35.584586" or "5:35.12"
        try:
            parts = td_str.strip().split(":")
            if len(parts) == 3:
                h = int(parts[0])
                m = int(parts[1])
                s = float(parts[2])
                return h * 3600 + m * 60 + s
            if len(parts) == 2:
                m = int(parts[0])
                s = float(parts[1])
                return m * 60 + s
            if len(parts) == 1:
                return float(parts[0])
        except Exception:
            return None
        return None

    def _format_eta(seconds: float | None) -> str:
        if seconds is None:
            return "计算中…"
        try:
            if seconds < 0:
                seconds = 0
            # Keep it compact: H:MM:SS (or D days, H:MM:SS if large)
            return str(datetime.timedelta(seconds=int(seconds)))
        except Exception:
            return "计算中…"

    try:
        total = int(float(total_epoch))
    except Exception:
        total = 0
    if total < 0:
        total = 0

    epoch = None
    pct_in_epoch = None

    last = None
    for last in _RE_TRAIN_EPOCH_PCT.finditer(log_text or ""):
        pass
    if last is not None:
        try:
            epoch = int(last.group(1))
            pct_in_epoch = int(last.group(2))
        except Exception:
            epoch = None
            pct_in_epoch = None

    last_done = None
    for last_done in _RE_EPOCH_DONE.finditer(log_text or ""):
        pass
    epoch_done = None
    if last_done is not None:
        try:
            epoch_done = int(last_done.group(1))
        except Exception:
            epoch_done = None

    # Parse recent epoch durations to estimate ETA
    elapsed_secs = []
    for m3 in _RE_EPOCH_ELAPSED.finditer(log_text or ""):
        try:
            e = int(m3.group(1))
            td = m3.group(2)
            sec = _parse_timedelta_seconds(td)
            if sec is not None and sec > 0:
                elapsed_secs.append((e, sec))
        except Exception:
            continue
    # Use last few epochs to smooth noise
    avg_epoch_sec = None
    if elapsed_secs:
        recent = [sec for _, sec in elapsed_secs[-3:]]
        if recent:
            avg_epoch_sec = sum(recent) / len(recent)

    # Prefer within-epoch percentage if available
    if epoch is not None and pct_in_epoch is not None:
        pct_in_epoch = max(0, min(100, pct_in_epoch))
        if total > 0:
            overall = ((epoch - 1) + pct_in_epoch / 100.0) / total * 100.0
            progress = int(max(0, min(100, round(overall))))
            # ETA: remaining part of current epoch + remaining full epochs
            eta = None
            if avg_epoch_sec is not None:
                eta = (1.0 - pct_in_epoch / 100.0) * avg_epoch_sec + max(
                    0, (total - epoch)
                ) * avg_epoch_sec
            status = f"Epoch {epoch}/{total}，本轮 {pct_in_epoch}% ，预计剩余 {_format_eta(eta)}"
        else:
            progress = pct_in_epoch
            status = f"Epoch {epoch}，本轮 {pct_in_epoch}% ，预计剩余 {_format_eta(None)}"
        return progress, status

    # Fallback: epoch done line
    if epoch_done is not None:
        if total > 0:
            progress = int(max(0, min(100, round(epoch_done / total * 100.0))))
            eta = None
            if avg_epoch_sec is not None:
                eta = max(0, (total - epoch_done)) * avg_epoch_sec
            status = f"已完成 Epoch {epoch_done}/{total} ，预计剩余 {_format_eta(eta)}"
        else:
            progress = 0
            status = f"已完成 Epoch {epoch_done} ，预计剩余 {_format_eta(None)}"
        return progress, status

    return 0, "等待训练日志输出…"


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    # Persist trainset_dir for future "history experiment" auto-fill
    _write_exp_meta(exp_dir, trainset_dir=trainset_dir)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    cmd = '"%s" infer/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f' % (
        config.python_cmd,
        trainset_dir,
        sr,
        n_p,
        now_dir,
        exp_dir,
        config.noparallel,
        config.preprocess_per,
    )
    logger.info("Execute: " + cmd)
    # , stdin=PIPE, stdout=PIPE,stderr=PIPE,cwd=now_dir
    p = Popen(cmd, shell=True)
    # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log


# but2.click(extract_f0,[gpus6,np7,f0method8,if_f0_3,trainset_dir4],[info2])
def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe):
    gpus = gpus.split("-")
    if config.device == "mps":
        gpus = [gpus[0]]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = (
                '"%s" infer/modules/train/extract/extract_f0_print.py "%s/logs/%s" %s %s'
                % (
                    config.python_cmd,
                    now_dir,
                    exp_dir,
                    n_p,
                    f0method,
                )
            )
            logger.info("Execute: " + cmd)
            p = Popen(
                cmd, shell=True, cwd=now_dir
            )  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
            # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
            done = [False]
            threading.Thread(
                target=if_done,
                args=(
                    done,
                    p,
                ),
            ).start()
        else:
            if gpus_rmvpe != "-":
                gpus_rmvpe = gpus_rmvpe.split("-")
                leng = len(gpus_rmvpe)
                ps = []
                for idx, n_g in enumerate(gpus_rmvpe):
                    cmd = (
                        '"%s" infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s/logs/%s" %s '
                        % (
                            config.python_cmd,
                            leng,
                            idx,
                            n_g,
                            now_dir,
                            exp_dir,
                            config.is_half,
                        )
                    )
                    logger.info("Execute: " + cmd)
                    p = Popen(
                        cmd, shell=True, cwd=now_dir
                    )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
                    ps.append(p)
                # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
                done = [False]
                threading.Thread(
                    target=if_done_multi,  #
                    args=(
                        done,
                        ps,
                    ),
                ).start()
            else:
                cmd = (
                    config.python_cmd
                    + ' infer/modules/train/extract/extract_f0_rmvpe_dml.py "%s/logs/%s" '
                    % (
                        now_dir,
                        exp_dir,
                    )
                )
                logger.info("Execute: " + cmd)
                p = Popen(
                    cmd, shell=True, cwd=now_dir
                )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
                p.wait()
                done = [True]
        while 1:
            with open(
                "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                yield (f.read())
            sleep(1)
            if done[0]:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        logger.info(log)
        yield log
    # 对不同part分别开多进程
    """
    n_part=int(sys.argv[1])
    i_part=int(sys.argv[2])
    i_gpu=sys.argv[3]
    exp_dir=sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)
    """
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
            '"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s %s'
            % (
                config.python_cmd,
                config.device,
                leng,
                idx,
                n_g,
                now_dir,
                exp_dir,
                version19,
                config.is_half,
            )
        )
        logger.info("Execute: " + cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log


def get_pretrained_models(path_str, f0_str, sr2):
    if_pretrained_generator_exist = os.access(
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warning(
            "assets/pretrained%s/%sG%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        logger.warning(
            "assets/pretrained%s/%sD%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    return (
        (
            "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_generator_exist
            else ""
        ),
        (
            "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_discriminator_exist
            else ""
        ),
    )


def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )


def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        {"visible": if_f0_3, "__type__": "update"},
        *get_pretrained_models(path_str, "f0" if if_f0_3 == True else "", sr2),
    )


# but3.click(click_train,[exp_dir1,sr2,if_f0_3,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16])
def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    log_interval,
    train_num_workers,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    try:
        # 生成filelist
        logger.info(
            "click_train called with: exp_dir1=%s, sr2=%s, version19=%s",
            exp_dir1,
            sr2,
            version19,
        )
        exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
        os.makedirs(exp_dir, exist_ok=True)
        gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
        feature_dir = (
            "%s/3_feature256" % (exp_dir)
            if version19 == "v1"
            else "%s/3_feature768" % (exp_dir)
        )

        # 前置产物检查：必须先“处理数据”和“特征提取”(带f0时还需要f0文件夹)
        missing_dirs = []
        required_dirs = [
            (gt_wavs_dir, "0_gt_wavs（处理数据后生成）"),
            (feature_dir, "3_feature（特征提取后生成）"),
        ]
        f0_dir = None
        f0nsf_dir = None
        if if_f0_3:
            f0_dir = "%s/2a_f0" % (exp_dir)
            f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
            required_dirs.extend(
                [
                    (f0_dir, "2a_f0（提取音高后生成）"),
                    (f0nsf_dir, "2b-f0nsf（提取音高后生成）"),
                ]
            )
        for p, desc in required_dirs:
            if not os.path.isdir(p):
                missing_dirs.append(f"{desc}: {p}")
        if missing_dirs:
            msg = (
                "无法开始训练：缺少训练前置产物目录。\n"
                + "\n".join(missing_dirs)
                + "\n\n请按顺序先执行：\n"
                + "1) 处理数据（生成 0_gt_wavs 等）\n"
                + "2) 特征提取（生成 3_feature；若模型带F0还会生成 2a_f0/2b-f0nsf）\n"
                + "完成后再点【训练模型】。"
            )
            logger.warning(msg)
            yield msg, 0, i18n("未开始")
            return

        # 目录存在但为空也要拦截（常见于步骤没跑完/被中断）
        if len(os.listdir(gt_wavs_dir)) == 0:
            yield f"无法开始训练：{gt_wavs_dir} 为空。请先执行【处理数据】。", 0, i18n("未开始")
            return
        if len(os.listdir(feature_dir)) == 0:
            yield f"无法开始训练：{feature_dir} 为空。请先执行【特征提取】。", 0, i18n("未开始")
            return
        if if_f0_3:
            if f0_dir and len(os.listdir(f0_dir)) == 0:
                yield (
                    f"无法开始训练：{f0_dir} 为空。请先执行【特征提取】(包含音高)。",
                    0,
                    i18n("未开始"),
                )
                return
            if f0nsf_dir and len(os.listdir(f0nsf_dir)) == 0:
                yield (
                    f"无法开始训练：{f0nsf_dir} 为空。请先执行【特征提取】(包含音高)。",
                    0,
                    i18n("未开始"),
                )
                return

        if if_f0_3:
            names = (
                set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
                & set([name.split(".")[0] for name in os.listdir(feature_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
            )
        else:
            names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
                [name.split(".")[0] for name in os.listdir(feature_dir)]
            )
        opt = []
        for name in names:
            if if_f0_3:
                opt.append(
                    "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                    % (
                        gt_wavs_dir.replace("\\", "\\\\"),
                        name,
                        feature_dir.replace("\\", "\\\\"),
                        name,
                        f0_dir.replace("\\", "\\\\"),
                        name,
                        f0nsf_dir.replace("\\", "\\\\"),
                        name,
                        spk_id5,
                    )
                )
            else:
                opt.append(
                    "%s/%s.wav|%s/%s.npy|%s"
                    % (
                        gt_wavs_dir.replace("\\", "\\\\"),
                        name,
                        feature_dir.replace("\\", "\\\\"),
                        name,
                        spk_id5,
                    )
                )
        fea_dim = 256 if version19 == "v1" else 768
        if if_f0_3:
            for _ in range(2):
                opt.append(
                    "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                    % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
                )
        else:
            for _ in range(2):
                opt.append(
                    "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                    % (now_dir, sr2, now_dir, fea_dim, spk_id5)
                )
        shuffle(opt)
        with open("%s/filelist.txt" % exp_dir, "w") as f:
            f.write("\n".join(opt))
        logger.debug("Write filelist done")

        # 训练日志文件：实时回显用（训练脚本也会往这里写）
        train_log_path = os.path.join(exp_dir, "train.log")
        try:
            open(train_log_path, "a", encoding="utf-8").close()
        except Exception:
            # fallback: do not block training if log file can't be touched
            pass

        logger.info("Use gpus: %s", str(gpus16))
        if pretrained_G14 == "":
            logger.info("No pretrained Generator")
        if pretrained_D15 == "":
            logger.info("No pretrained Discriminator")

        if version19 == "v1" or sr2 == "40k":
            config_path = "v1/%s.json" % sr2
        else:
            config_path = "v2/%s.json" % sr2
        config_save_path = os.path.join(exp_dir, "config.json")
        if not pathlib.Path(config_save_path).exists():
            with open(config_save_path, "w", encoding="utf-8") as f:
                # 深拷贝一份模板，避免意外修改全局模板
                cfg = json.loads(json.dumps(config.json_config[config_path]))
                # 默认日志输出频率：优先使用模板值；若缺失则给一个较省的默认值
                cfg.setdefault("train", {}).setdefault("log_interval", 50)
                json.dump(
                    cfg,
                    f,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                )
                f.write("\n")
        # Always apply UI-selected log interval to experiment config
        try:
            log_interval_int = int(log_interval)
            if log_interval_int < 1:
                log_interval_int = 1
        except Exception:
            log_interval_int = 50
        try:
            with open(config_save_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg.setdefault("train", {})["log_interval"] = log_interval_int
            with open(config_save_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=4, sort_keys=True)
                f.write("\n")
        except Exception as e:
            logger.warning("Failed to update log_interval in %s: %s", config_save_path, e)

        # DataLoader workers (passed to training script via env var)
        try:
            train_num_workers_int = int(train_num_workers)
            if train_num_workers_int < 0:
                train_num_workers_int = 0
        except Exception:
            train_num_workers_int = 0

        if gpus16:
            cmd = (
                '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
                % (
                    config.python_cmd,
                    exp_dir1,
                    sr2,
                    1 if if_f0_3 else 0,
                    batch_size12,
                    gpus16,
                    total_epoch11,
                    save_epoch10,
                    "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                    "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                    1 if if_save_latest13 == i18n("是") else 0,
                    1 if if_cache_gpu17 == i18n("是") else 0,
                    1 if if_save_every_weights18 == i18n("是") else 0,
                    version19,
                )
            )
        else:
            cmd = (
                '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
                % (
                    config.python_cmd,
                    exp_dir1,
                    sr2,
                    1 if if_f0_3 else 0,
                    batch_size12,
                    total_epoch11,
                    save_epoch10,
                    "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                    "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                    1 if if_save_latest13 == i18n("是") else 0,
                    1 if if_cache_gpu17 == i18n("是") else 0,
                    1 if if_save_every_weights18 == i18n("是") else 0,
                    version19,
                )
            )
        logger.info("Execute: %s", cmd)
        env = os.environ.copy()
        env["RVC_TRAIN_NUM_WORKERS"] = str(train_num_workers_int)
        popen_kwargs = {"shell": True, "cwd": now_dir, "env": env}
        # Start new session so we can pause/resume the whole group (posix)
        if platform.system() != "Windows":
            popen_kwargs["start_new_session"] = True
        p = Popen(cmd, **popen_kwargs)
        _set_train_proc(p, exp_dir1)

        events_path = os.path.join(exp_dir, "train_events.jsonl")
        tailer = JsonlTailer(events_path)
        try:
            total_epoch_int = int(float(total_epoch11))
        except Exception:
            total_epoch_int = 0
        monitor = TrainMonitor(total_epoch=total_epoch_int, avg_window=50)
        while p.poll() is None:
            sleep(1)
            log_tail = tail_text_file(train_log_path)
            new_events = tailer.read_new()
            if new_events:
                monitor.update(new_events)
            rendered = monitor.render(stall_threshold_sec=60)

            # Fallback: if no structured events yet, use legacy log parsing to show *something*
            if not monitor.key_events and log_tail.strip():
                progress, status = parse_train_progress(log_tail, total_epoch11)
                rendered["state"] = status
                rendered["epoch_pct"] = progress

            summary = rendered["summary"] or ("\n".join(log_tail.splitlines()[-30:]) if log_tail else "")
            state_str = str(rendered.get("state") or "")
            # If paused, override badge display and avoid stall false-positive
            if _TRAIN_PROC.get("paused"):
                state_str = "已暂停 | " + state_str
                rendered["diag"] = ""
            time_html, speed_html, loss_html = _render_train_cards_html(state_str, rendered)

            # New panel outputs + legacy hidden outputs
            yield (
                state_str,
                int(rendered["epoch_pct"]),
                int(rendered["step_pct"]),
                time_html,
                speed_html,
                loss_html,
                rendered["diag"],
                rendered["events"],
                summary,
                log_tail,
                summary,  # legacy info3
                int(rendered["epoch_pct"]),  # legacy progress
                state_str,  # legacy status
            )
        rc = p.wait()
        _clear_train_proc()
        # Final refresh (events/log tail)
        new_events = tailer.read_new()
        if new_events:
            monitor.update(new_events)
        rendered = monitor.render(stall_threshold_sec=60)
        log_tail = tail_text_file(train_log_path)
        summary = rendered["summary"] or ("\n".join(log_tail.splitlines()[-30:]) if log_tail else "")
        time_html, speed_html, loss_html = _render_train_cards_html(
            str(rendered.get("state") or ""), rendered
        )

        if rc not in (0, 149):
            diag = (rendered["diag"] + "\n\n" if rendered["diag"] else "") + f"训练进程异常退出(exit code={rc})"
            state = f"异常退出 (code={rc})"
            yield (
                state,
                int(rendered["epoch_pct"]),
                int(rendered["step_pct"]),
                time_html,
                speed_html,
                loss_html,
                diag,
                rendered["events"],
                summary,
                log_tail,
                summary,
                int(rendered["epoch_pct"]),
                state,
            )
        else:
            state = i18n("已完成")
            yield (
                state,
                100,
                100,
                time_html,
                speed_html,
                loss_html,
                "",
                rendered["events"],
                summary,
                log_tail,
                summary,
                100,
                state,
            )
    except Exception:
        err = traceback.format_exc()
        logger.error(err)
        _clear_train_proc()
        yield (
            i18n("异常"),
            0,
            0,
            "—",
            "—",
            "—",
            err,
            "",
            "",
            "",
            err,
            0,
            i18n("异常"),
        )


# but4.click(train_index, [exp_dir1], info3)
def train_index(exp_dir1, version19):
    # 延迟导入：faiss / sklearn 在部分 macOS 环境下可能导致启动时崩溃
    try:
        import faiss  # type: ignore
    except Exception as e:
        return f"缺少 faiss 或 faiss 导入失败：{e}\n请先安装 faiss 后再训练索引。"
    try:
        from sklearn.cluster import MiniBatchKMeans  # type: ignore
    except Exception as e:
        return f"缺少 scikit-learn 或导入失败：{e}\n请先安装 scikit-learn 后再训练索引。"

    # exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    exp_dir = "logs/%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if not os.path.exists(feature_dir):
        return "请先进行特征提取!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "请先进行特征提取！"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)

    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append(
        "成功构建索引 added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    try:
        if outside_index_root:
            os.makedirs(outside_index_root, exist_ok=True)
            src = "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index" % (
                exp_dir,
                n_ivf,
                index_ivf.nprobe,
                exp_dir1,
                version19,
            )
            dst = "%s/%s_IVF%s_Flat_nprobe_%s_%s_%s.index" % (
                outside_index_root,
                exp_dir1,
                n_ivf,
                index_ivf.nprobe,
                exp_dir1,
                version19,
            )
            # If existing file/link conflicts, remove then recreate
            if os.path.lexists(dst):
                try:
                    os.remove(dst)
                except Exception:
                    pass
            # Windows uses hardlink by default in upstream; fallback to symlink if cross-device
            if platform.system() == "Windows":
                try:
                    os.link(src, dst)
                except Exception:
                    os.symlink(src, dst)
            else:
                os.symlink(src, dst)
            infos.append("链接索引到外部-%s" % (outside_index_root))
        else:
            infos.append("未配置 outside_index_root，跳过链接索引到外部")
    except Exception as e:
        infos.append("链接索引到外部-%s失败：%s" % (outside_index_root, str(e)))

    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    # infos.append("成功构建索引，added_IVF%s_Flat_FastScan_%s.index"%(n_ivf,version19))
    yield "\n".join(infos)


def train_index_ui(exp_dir1, version19):
    """
    UI wrapper: returns (log, progress, status) for gradio outputs.
    """
    for msg in train_index(exp_dir1, version19):
        state = i18n("索引训练")
        # basic cards for non-step tasks
        time_html, speed_html, loss_html = _render_train_cards_html(state, {"elapsed_sec": None, "eta_sec": None, "eta_note": "", "avg_step_sec": None, "last_step_sec": None, "losses": {}})
        yield (
            state,  # state
            100,  # epoch pct
            0,  # step pct
            time_html,  # time
            speed_html,  # speed
            loss_html,  # loss
            "",  # diag
            state,  # events
            msg,  # summary
            msg,  # raw log
            msg,  # legacy info3
            100,  # legacy progress
            state,  # legacy status
        )


# but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17], info3)
def train1key(
    exp_dir1,
    sr2,
    if_f0_3,
    trainset_dir4,
    spk_id5,
    np7,
    f0method8,
    save_epoch10,
    total_epoch11,
    batch_size12,
    log_interval,
    train_num_workers,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    gpus_rmvpe,
):
    # step1:处理数据
    msg = i18n("step1:正在处理数据")
    time_html, speed_html, loss_html = _render_train_cards_html(i18n("数据处理"), {"elapsed_sec": None, "eta_sec": None, "eta_note": "", "avg_step_sec": None, "last_step_sec": None, "losses": {}})
    yield (i18n("数据处理"), 0, 0, time_html, speed_html, loss_html, "", msg, msg, msg, msg, 0, i18n("数据处理"))
    for log in preprocess_dataset(trainset_dir4, exp_dir1, sr2, np7):
        time_html, speed_html, loss_html = _render_train_cards_html(i18n("数据处理"), {"elapsed_sec": None, "eta_sec": None, "eta_note": "", "avg_step_sec": None, "last_step_sec": None, "losses": {}})
        yield (i18n("数据处理"), 0, 0, time_html, speed_html, loss_html, "", "", log, log, log, 0, i18n("数据处理"))

    # step2a:提取音高
    msg = i18n("step2:正在提取音高&正在提取特征")
    time_html, speed_html, loss_html = _render_train_cards_html(i18n("特征提取"), {"elapsed_sec": None, "eta_sec": None, "eta_note": "", "avg_step_sec": None, "last_step_sec": None, "losses": {}})
    yield (i18n("特征提取"), 0, 0, time_html, speed_html, loss_html, "", msg, msg, msg, msg, 0, i18n("特征提取"))
    for log in extract_f0_feature(
        gpus16, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe
    ):
        time_html, speed_html, loss_html = _render_train_cards_html(i18n("特征提取"), {"elapsed_sec": None, "eta_sec": None, "eta_note": "", "avg_step_sec": None, "last_step_sec": None, "losses": {}})
        yield (i18n("特征提取"), 0, 0, time_html, speed_html, loss_html, "", "", log, log, log, 0, i18n("特征提取"))

    # step3a:训练模型
    msg = i18n("step3a:正在训练模型")
    yield (i18n("训练中"), 0, 0, "—", "—", "—", "", msg, msg, msg, msg, 0, i18n("训练中"))
    for out in click_train(
        exp_dir1,
        sr2,
        if_f0_3,
        spk_id5,
        save_epoch10,
        total_epoch11,
        batch_size12,
        log_interval,
        train_num_workers,
        if_save_latest13,
        pretrained_G14,
        pretrained_D15,
        gpus16,
        if_cache_gpu17,
        if_save_every_weights18,
        version19,
    ):
        yield out

    # step3b:训练索引
    msg = i18n("step3b:正在训练索引")
    time_html, speed_html, loss_html = _render_train_cards_html(i18n("索引训练"), {"elapsed_sec": None, "eta_sec": None, "eta_note": "", "avg_step_sec": None, "last_step_sec": None, "losses": {}})
    yield (i18n("索引训练"), 100, 0, time_html, speed_html, loss_html, "", msg, msg, msg, msg, 100, i18n("索引训练"))
    for out in train_index_ui(exp_dir1, version19):
        yield out
    done_msg = i18n("全流程结束！")
    time_html, speed_html, loss_html = _render_train_cards_html(i18n("已完成"), {"elapsed_sec": None, "eta_sec": None, "eta_note": "", "avg_step_sec": None, "last_step_sec": None, "losses": {}})
    yield (i18n("已完成"), 100, 100, time_html, speed_html, loss_html, "", done_msg, done_msg, done_msg, done_msg, 100, i18n("已完成"))


#                    ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__])
def change_info_(ckpt_path):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(
            ckpt_path.replace(os.path.basename(ckpt_path), "train.log"), "r"
        ) as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


F0GPUVisible = config.dml == False


def change_f0_method(f0method8):
    if f0method8 == "rmvpe_gpu":
        visible = F0GPUVisible
    else:
        visible = False
    return {"visible": visible, "__type__": "update"}


with gr.Blocks(title="RVC WebUI") as app:
    gr.Markdown("## RVC WebUI")
    gr.Markdown(
        value=i18n(
            "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>."
        )
    )
    with gr.Tabs():
        with gr.TabItem(i18n("模型推理")):
            with gr.Row():
                sid0 = gr.Dropdown(label=i18n("推理音色"), choices=sorted(names))
                with gr.Column():
                    refresh_button = gr.Button(
                        i18n("刷新音色列表和索引路径"), variant="primary"
                    )
                    clean_button = gr.Button(i18n("卸载音色省显存"), variant="primary")
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label=i18n("请选择说话人id"),
                    value=0,
                    visible=False,
                    interactive=True,
                )
                clean_button.click(
                    fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean"
                )
            with gr.TabItem(i18n("单次推理")):
                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            vc_transform0 = gr.Number(
                                label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"),
                                value=0,
                            )
                            input_audio0 = gr.Textbox(
                                label=i18n(
                                    "输入待处理音频文件路径(默认是正确格式示例)"
                                ),
                                placeholder="C:\\Users\\Desktop\\audio_example.wav",
                            )
                            file_index1 = gr.Textbox(
                                label=i18n(
                                    "特征检索库文件路径,为空则使用下拉的选择结果"
                                ),
                                placeholder="C:\\Users\\Desktop\\model_example.index",
                                interactive=True,
                            )
                            file_index2 = gr.Dropdown(
                                label=i18n("自动检测index路径,下拉式选择(dropdown)"),
                                choices=sorted(index_paths),
                                interactive=True,
                            )
                            f0method0 = gr.Radio(
                                label=i18n(
                                    "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU"
                                ),
                                choices=(
                                    ["pm", "harvest", "crepe", "rmvpe"]
                                    if config.dml == False
                                    else ["pm", "harvest", "rmvpe"]
                                ),
                                value="rmvpe",
                                interactive=True,
                            )

                        with gr.Column():
                            resample_sr0 = gr.Slider(
                                minimum=0,
                                maximum=48000,
                                label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                                value=0,
                                step=1,
                                interactive=True,
                            )
                            rms_mix_rate0 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=i18n(
                                    "输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"
                                ),
                                value=0.25,
                                interactive=True,
                            )
                            protect0 = gr.Slider(
                                minimum=0,
                                maximum=0.5,
                                label=i18n(
                                    "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                                ),
                                value=0.33,
                                step=0.01,
                                interactive=True,
                            )
                            filter_radius0 = gr.Slider(
                                minimum=0,
                                maximum=7,
                                label=i18n(
                                    ">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"
                                ),
                                value=3,
                                step=1,
                                interactive=True,
                            )
                            index_rate1 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=i18n("检索特征占比"),
                                value=0.75,
                                interactive=True,
                            )
                            f0_file = gr.File(
                                label=i18n(
                                    "F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调"
                                ),
                                visible=False,
                            )

                            refresh_button.click(
                                fn=change_choices,
                                inputs=[],
                                outputs=[sid0, file_index2],
                                api_name="infer_refresh",
                            )
                            # file_big_npy1 = gr.Textbox(
                            #     label=i18n("特征文件路径"),
                            #     value="E:\\codes\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                            #     interactive=True,
                            # )
                with gr.Group():
                    with gr.Column():
                        but0 = gr.Button(i18n("转换"), variant="primary")
                        with gr.Row():
                            vc_output1 = gr.Textbox(label=i18n("输出信息"))
                            vc_output2 = gr.Audio(
                                label=i18n("输出音频(右下角三个点,点了可以下载)")
                            )

                        but0.click(
                            vc.vc_single,
                            [
                                spk_item,
                                input_audio0,
                                vc_transform0,
                                f0_file,
                                f0method0,
                                file_index1,
                                file_index2,
                                # file_big_npy1,
                                index_rate1,
                                filter_radius0,
                                resample_sr0,
                                rms_mix_rate0,
                                protect0,
                            ],
                            [vc_output1, vc_output2],
                            api_name="infer_convert",
                        )
            with gr.TabItem(i18n("批量推理")):
                gr.Markdown(
                    value=i18n(
                        "批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. "
                    )
                )
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Number(
                            label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"),
                            value=0,
                        )
                        opt_input = gr.Textbox(
                            label=i18n("指定输出文件夹"), value="opt"
                        )
                        file_index3 = gr.Textbox(
                            label=i18n("特征检索库文件路径,为空则使用下拉的选择结果"),
                            value="",
                            interactive=True,
                        )
                        file_index4 = gr.Dropdown(
                            label=i18n("自动检测index路径,下拉式选择(dropdown)"),
                            choices=sorted(index_paths),
                            interactive=True,
                        )
                        f0method1 = gr.Radio(
                            label=i18n(
                                "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU"
                            ),
                            choices=(
                                ["pm", "harvest", "crepe", "rmvpe"]
                                if config.dml == False
                                else ["pm", "harvest", "rmvpe"]
                            ),
                            value="rmvpe",
                            interactive=True,
                        )
                        format1 = gr.Radio(
                            label=i18n("导出文件格式"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="wav",
                            interactive=True,
                        )

                        refresh_button.click(
                            fn=lambda: change_choices()[1],
                            inputs=[],
                            outputs=file_index4,
                            api_name="infer_refresh_batch",
                        )
                        # file_big_npy2 = gr.Textbox(
                        #     label=i18n("特征文件路径"),
                        #     value="E:\\codes\\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                        #     interactive=True,
                        # )

                    with gr.Column():
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n(
                                "输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"
                            ),
                            value=1,
                            interactive=True,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                            ),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                        filter_radius1 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(
                                ">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"
                            ),
                            value=3,
                            step=1,
                            interactive=True,
                        )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("检索特征占比"),
                            value=1,
                            interactive=True,
                        )
                with gr.Row():
                    dir_input = gr.Textbox(
                        label=i18n(
                            "输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)"
                        ),
                        placeholder="C:\\Users\\Desktop\\input_vocal_dir",
                    )
                    inputs = gr.File(
                        file_count="multiple",
                        label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹"),
                    )

                with gr.Row():
                    but1 = gr.Button(i18n("转换"), variant="primary")
                    vc_output3 = gr.Textbox(label=i18n("输出信息"))

                    but1.click(
                        vc.vc_multi,
                        [
                            spk_item,
                            dir_input,
                            opt_input,
                            inputs,
                            vc_transform1,
                            f0method1,
                            file_index3,
                            file_index4,
                            # file_big_npy2,
                            index_rate2,
                            filter_radius1,
                            resample_sr1,
                            rms_mix_rate1,
                            protect1,
                            format1,
                        ],
                        [vc_output3],
                        api_name="infer_convert_batch",
                    )
                sid0.change(
                    fn=vc.get_vc,
                    inputs=[sid0, protect0, protect1],
                    outputs=[spk_item, protect0, protect1, file_index2, file_index4],
                    api_name="infer_change_voice",
                )
        with gr.TabItem(i18n("伴奏人声分离&去混响&去回声")):
            with gr.Group():
                gr.Markdown(
                    value=i18n(
                        "人声伴奏分离批量处理， 使用UVR5模型。 <br>合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>模型分为三类： <br>1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> 3、去混响、去延迟模型（by FoxJoy）：<br>  (1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>去混响/去延迟，附：<br>1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>2、MDX-Net-Dereverb模型挺慢的；<br>3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。"
                    )
                )
                with gr.Row():
                    with gr.Column():
                        dir_wav_input = gr.Textbox(
                            label=i18n("输入待处理音频文件夹路径"),
                            placeholder="C:\\Users\\Desktop\\todo-songs",
                        )
                        wav_inputs = gr.File(
                            file_count="multiple",
                            label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹"),
                        )
                    with gr.Column():
                        model_choose = gr.Dropdown(
                            label=i18n("模型"), choices=uvr5_names
                        )
                        agg = gr.Slider(
                            minimum=0,
                            maximum=20,
                            step=1,
                            label="人声提取激进程度",
                            value=10,
                            interactive=True,
                            visible=False,  # 先不开放调整
                        )
                        opt_vocal_root = gr.Textbox(
                            label=i18n("指定输出主人声文件夹"), value="opt"
                        )
                        opt_ins_root = gr.Textbox(
                            label=i18n("指定输出非主人声文件夹"), value="opt"
                        )
                        format0 = gr.Radio(
                            label=i18n("导出文件格式"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="flac",
                            interactive=True,
                        )
                    but2 = gr.Button(i18n("转换"), variant="primary")
                    vc_output4 = gr.Textbox(label=i18n("输出信息"))
                    but2.click(
                        uvr,
                        [
                            model_choose,
                            dir_wav_input,
                            opt_vocal_root,
                            wav_inputs,
                            opt_ins_root,
                            agg,
                            format0,
                        ],
                        [vc_output4],
                        api_name="uvr_convert",
                    )
        with gr.TabItem(i18n("训练")):
            gr.Markdown(
                value=i18n(
                    "step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. "
                )
            )
            with gr.Row():
                exp_history = gr.Dropdown(
                    label=i18n("历史实验"),
                    choices=_safe_list_log_experiments(),
                    value=None,
                    interactive=True,
                )
                exp_history_refresh = gr.Button(i18n("刷新实验列表"))
                exp_history_apply = gr.Button(i18n("加载并应用到表单"), variant="primary")
                exp_history_info = gr.Textbox(
                    label=i18n("提示"),
                    value=i18n("可从历史实验自动回填训练集路径/采样率/版本/是否F0。"),
                    max_lines=4,
                    interactive=False,
                )
            with gr.Row():
                exp_dir1 = gr.Textbox(label=i18n("输入实验名"), value="mi-test")
                sr2 = gr.Radio(
                    label=i18n("目标采样率"),
                    choices=["40k", "48k"],
                    value="40k",
                    interactive=True,
                )
                if_f0_3 = gr.Radio(
                    label=i18n("模型是否带音高指导(唱歌一定要, 语音可以不要)"),
                    choices=[True, False],
                    value=True,
                    interactive=True,
                )
                version19 = gr.Radio(
                    label=i18n("版本"),
                    choices=["v1", "v2"],
                    value="v2",
                    interactive=True,
                    visible=True,
                )
                np7 = gr.Slider(
                    minimum=0,
                    maximum=config.n_cpu,
                    step=1,
                    label=i18n("提取音高和处理数据使用的CPU进程数"),
                    value=int(np.ceil(config.n_cpu / 1.5)),
                    interactive=True,
                )
            with gr.Group():  # 暂时单人的, 后面支持最多4人的#数据处理
                gr.Markdown(
                    value=i18n(
                        "step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. "
                    )
                )
                with gr.Row():
                    trainset_dir4 = gr.Textbox(
                        label=i18n("输入训练文件夹路径"),
                        value=i18n("E:\\语音音频+标注\\米津玄师\\src"),
                    )
                    trainset_dir_browse = gr.Button(i18n("选择目录…"))
                    spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label=i18n("请指定说话人id"),
                        value=0,
                        interactive=True,
                    )
                    but1 = gr.Button(i18n("处理数据"), variant="primary")
                    info1 = gr.Textbox(label=i18n("输出信息"), value="")
                    trainset_dir_browse.click(
                        fn=_browse_directory,
                        inputs=[trainset_dir4],
                        outputs=[trainset_dir4],
                    )
                    but1.click(
                        preprocess_dataset,
                        [trainset_dir4, exp_dir1, sr2, np7],
                        [info1],
                        api_name="train_preprocess",
                    )
            exp_history_refresh.click(fn=_refresh_exp_choices, inputs=[], outputs=[exp_history])
            with gr.Group():
                gr.Markdown(
                    value=i18n(
                        "step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)"
                    )
                )
                with gr.Row():
                    with gr.Column():
                        gpus6 = gr.Textbox(
                            label=i18n(
                                "以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"
                            ),
                            value=gpus,
                            interactive=True,
                            visible=F0GPUVisible,
                        )
                        gpu_info9 = gr.Textbox(
                            label=i18n("显卡信息"), value=gpu_info, visible=F0GPUVisible
                        )
                    with gr.Column():
                        f0method8 = gr.Radio(
                            label=i18n(
                                "选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢,rmvpe效果最好且微吃CPU/GPU"
                            ),
                            choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
                            value="rmvpe_gpu",
                            interactive=True,
                        )
                        gpus_rmvpe = gr.Textbox(
                            label=i18n(
                                "rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡0上跑2个进程并在卡1上跑1个进程"
                            ),
                            value="%s-%s" % (gpus, gpus),
                            interactive=True,
                            visible=F0GPUVisible,
                        )
                    but2 = gr.Button(i18n("特征提取"), variant="primary")
                    info2 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                    f0method8.change(
                        fn=change_f0_method,
                        inputs=[f0method8],
                        outputs=[gpus_rmvpe],
                    )
                    but2.click(
                        extract_f0_feature,
                        [
                            gpus6,
                            np7,
                            f0method8,
                            if_f0_3,
                            exp_dir1,
                            version19,
                            gpus_rmvpe,
                        ],
                        [info2],
                        api_name="train_extract_f0_feature",
                    )
            with gr.Group():
                gr.Markdown(value=i18n("step3: 填写训练设置, 开始训练模型和索引"))
                with gr.Row():
                    save_epoch10 = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        label=i18n("保存频率save_every_epoch"),
                        value=5,
                        interactive=True,
                    )
                    total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=1000,
                        step=1,
                        label=i18n("总训练轮数total_epoch"),
                        value=20,
                        interactive=True,
                    )
                    batch_size12 = gr.Slider(
                        minimum=1,
                        maximum=40,
                        step=1,
                        label=i18n("每张显卡的batch_size"),
                        value=default_batch_size,
                        interactive=True,
                    )
                    log_interval_train = gr.Slider(
                        minimum=1,
                        maximum=500,
                        step=1,
                        label=i18n("日志频率log_interval（每N step写一次日志/图像）"),
                        value=50,
                        interactive=True,
                    )
                    train_num_workers_ui = gr.Slider(
                        minimum=0,
                        maximum=int(config.n_cpu),
                        step=1,
                        label=i18n("DataLoader workers（0更稳，2/4通常更快）"),
                        value=min(4, int(config.n_cpu)),
                        interactive=True,
                    )
                    if_save_latest13 = gr.Radio(
                        label=i18n("是否仅保存最新的ckpt文件以节省硬盘空间"),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("否"),
                        interactive=True,
                    )
                    if_cache_gpu17 = gr.Radio(
                        label=i18n(
                            "是否缓存所有训练集至显存. 10min以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速"
                        ),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("否"),
                        interactive=True,
                    )
                    if_save_every_weights18 = gr.Radio(
                        label=i18n(
                            "是否在每次保存时间点将最终小模型保存至weights文件夹"
                        ),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("否"),
                        interactive=True,
                    )
                with gr.Row():
                    pretrained_G14 = gr.Textbox(
                        label=i18n("加载预训练底模G路径"),
                        value="assets/pretrained_v2/f0G40k.pth",
                        interactive=True,
                    )
                    pretrained_D15 = gr.Textbox(
                        label=i18n("加载预训练底模D路径"),
                        value="assets/pretrained_v2/f0D40k.pth",
                        interactive=True,
                    )
                    sr2.change(
                        change_sr2,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15],
                    )
                    version19.change(
                        change_version19,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15, sr2],
                    )
                    if_f0_3.change(
                        change_f0,
                        [if_f0_3, sr2, version19],
                        [f0method8, gpus_rmvpe, pretrained_G14, pretrained_D15],
                    )
                    gpus16 = gr.Textbox(
                        label=i18n(
                            "以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"
                        ),
                        value=gpus,
                        interactive=True,
                    )
                    but3 = gr.Button(i18n("训练模型"), variant="primary")
                    but4 = gr.Button(i18n("训练特征索引"), variant="primary")
                    but5 = gr.Button(i18n("一键训练"), variant="primary")
                    # New training monitor panel (A+B)
                    train_state = gr.Textbox(
                        label=i18n("状态"),
                        value=i18n("未开始"),
                        max_lines=1,
                        interactive=False,
                    )
                    with gr.Row():
                        train_pause_btn = gr.Button(i18n("暂停训练"))
                        train_resume_btn = gr.Button(i18n("继续训练"))
                        train_stop_btn = gr.Button(i18n("停止训练"))
                        train_proc_refresh = gr.Button(i18n("刷新运行状态"))
                    with gr.Row():
                        train_epoch_progress = gr.Slider(
                            minimum=0,
                            maximum=100,
                            step=1,
                            label=i18n("总体进度（按epoch/step估算）"),
                            value=0,
                            interactive=False,
                        )
                        train_step_progress = gr.Slider(
                            minimum=0,
                            maximum=100,
                            step=1,
                            label=i18n("本epoch进度（若可用）"),
                            value=0,
                            interactive=False,
                        )
                    train_time = gr.HTML(value="")
                    train_speed = gr.HTML(value="")
                    train_loss = gr.HTML(value="")
                    with gr.Accordion(i18n("诊断与日志"), open=False):
                        train_control_msg = gr.Textbox(
                            label=i18n("控制操作"),
                            value="",
                            max_lines=2,
                            interactive=False,
                        )
                        train_proc_status = gr.Textbox(
                            label=i18n("训练进程状态"),
                            value="",
                            max_lines=6,
                            interactive=False,
                        )
                        train_diag = gr.Textbox(
                            label=i18n("诊断"),
                            value="",
                            max_lines=6,
                            interactive=False,
                        )
                        train_events = gr.Textbox(
                            label=i18n("关键事件"),
                            value="",
                            max_lines=8,
                            interactive=False,
                        )
                        with gr.Tabs():
                            with gr.TabItem(i18n("训练摘要")):
                                train_log_summary = gr.Textbox(
                                    label=i18n("摘要"),
                                    value="",
                                    max_lines=18,
                                    interactive=False,
                                )
                            with gr.TabItem(i18n("原始日志 tail")):
                                train_log_raw = gr.Textbox(
                                    label=i18n("train.log"),
                                    value="",
                                    max_lines=18,
                                    interactive=False,
                                )

                    train_pause_btn.click(fn=pause_training_ui, inputs=[], outputs=[train_control_msg])
                    train_resume_btn.click(fn=resume_training_ui, inputs=[], outputs=[train_control_msg])
                    train_stop_btn.click(fn=stop_training_ui, inputs=[], outputs=[train_control_msg])
                    train_proc_refresh.click(fn=get_train_runtime_status_ui, inputs=[exp_dir1], outputs=[train_proc_status])

                    # Legacy outputs (hidden): keep compatibility with older callbacks
                    info3 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=10, visible=False)
                    train_progress = gr.Slider(
                        minimum=0, maximum=100, step=1, label=i18n("训练进度(%)"), value=0, interactive=False, visible=False
                    )
                    train_status = gr.Textbox(
                        label=i18n("训练状态"), value=i18n("未开始"), max_lines=1, interactive=False, visible=False
                    )

                    # Apply selected history experiment to form + snapshots (now that all outputs exist)
                    exp_history_apply.click(
                        fn=_apply_exp_selection,
                        inputs=[exp_history],
                        outputs=[
                            exp_dir1,
                            trainset_dir4,
                            sr2,
                            if_f0_3,
                            version19,
                            pretrained_G14,
                            pretrained_D15,
                            exp_history_info,
                            info1,
                            info2,
                            save_epoch10,
                            total_epoch11,
                            batch_size12,
                            log_interval_train,
                            gpus16,
                            if_save_latest13,
                            if_cache_gpu17,
                            if_save_every_weights18,
                            train_proc_status,
                            train_state,
                            train_epoch_progress,
                            train_step_progress,
                            train_time,
                            train_speed,
                            train_loss,
                            train_diag,
                            train_events,
                            train_log_summary,
                            train_log_raw,
                            info3,
                            train_progress,
                            train_status,
                        ],
                    )
                    but3.click(
                        click_train,
                        [
                            exp_dir1,
                            sr2,
                            if_f0_3,
                            spk_id5,
                            save_epoch10,
                            total_epoch11,
                            batch_size12,
                            log_interval_train,
                            train_num_workers_ui,
                            if_save_latest13,
                            pretrained_G14,
                            pretrained_D15,
                            gpus16,
                            if_cache_gpu17,
                            if_save_every_weights18,
                            version19,
                        ],
                        [
                            train_state,
                            train_epoch_progress,
                            train_step_progress,
                            train_time,
                            train_speed,
                            train_loss,
                            train_diag,
                            train_events,
                            train_log_summary,
                            train_log_raw,
                            info3,
                            train_progress,
                            train_status,
                        ],
                        api_name="train_start",
                    )
                    but4.click(
                        train_index_ui,
                        [exp_dir1, version19],
                        [
                            train_state,
                            train_epoch_progress,
                            train_step_progress,
                            train_time,
                            train_speed,
                            train_loss,
                            train_diag,
                            train_events,
                            train_log_summary,
                            train_log_raw,
                            info3,
                            train_progress,
                            train_status,
                        ],
                    )
                    but5.click(
                        train1key,
                        [
                            exp_dir1,
                            sr2,
                            if_f0_3,
                            trainset_dir4,
                            spk_id5,
                            np7,
                            f0method8,
                            save_epoch10,
                            total_epoch11,
                            batch_size12,
                            log_interval_train,
                            train_num_workers_ui,
                            if_save_latest13,
                            pretrained_G14,
                            pretrained_D15,
                            gpus16,
                            if_cache_gpu17,
                            if_save_every_weights18,
                            version19,
                            gpus_rmvpe,
                        ],
                        [
                            train_state,
                            train_epoch_progress,
                            train_step_progress,
                            train_time,
                            train_speed,
                            train_loss,
                            train_diag,
                            train_events,
                            train_log_summary,
                            train_log_raw,
                            info3,
                            train_progress,
                            train_status,
                        ],
                        api_name="train_start_all",
                    )

        with gr.TabItem(i18n("ckpt处理")):
            with gr.Group():
                gr.Markdown(value=i18n("模型融合, 可用于测试音色融合"))
                with gr.Row():
                    ckpt_a = gr.Textbox(
                        label=i18n("A模型路径"), value="", interactive=True
                    )
                    ckpt_b = gr.Textbox(
                        label=i18n("B模型路径"), value="", interactive=True
                    )
                    alpha_a = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("A模型权重"),
                        value=0.5,
                        interactive=True,
                    )
                with gr.Row():
                    sr_ = gr.Radio(
                        label=i18n("目标采样率"),
                        choices=["40k", "48k"],
                        value="40k",
                        interactive=True,
                    )
                    if_f0_ = gr.Radio(
                        label=i18n("模型是否带音高指导"),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("是"),
                        interactive=True,
                    )
                    info__ = gr.Textbox(
                        label=i18n("要置入的模型信息"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    name_to_save0 = gr.Textbox(
                        label=i18n("保存的模型名不带后缀"),
                        value="",
                        max_lines=1,
                        interactive=True,
                    )
                    version_2 = gr.Radio(
                        label=i18n("模型版本型号"),
                        choices=["v1", "v2"],
                        value="v1",
                        interactive=True,
                    )
                with gr.Row():
                    but6 = gr.Button(i18n("融合"), variant="primary")
                    info4 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                but6.click(
                    merge,
                    [
                        ckpt_a,
                        ckpt_b,
                        alpha_a,
                        sr_,
                        if_f0_,
                        info__,
                        name_to_save0,
                        version_2,
                    ],
                    info4,
                    api_name="ckpt_merge",
                )  # def merge(path1,path2,alpha1,sr,f0,info):
            with gr.Group():
                gr.Markdown(
                    value=i18n("修改模型信息(仅支持weights文件夹下提取的小模型文件)")
                )
                with gr.Row():
                    ckpt_path0 = gr.Textbox(
                        label=i18n("模型路径"), value="", interactive=True
                    )
                    info_ = gr.Textbox(
                        label=i18n("要改的模型信息"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    name_to_save1 = gr.Textbox(
                        label=i18n("保存的文件名, 默认空为和源文件同名"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                with gr.Row():
                    but7 = gr.Button(i18n("修改"), variant="primary")
                    info5 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                but7.click(
                    change_info,
                    [ckpt_path0, info_, name_to_save1],
                    info5,
                    api_name="ckpt_modify",
                )
            with gr.Group():
                gr.Markdown(
                    value=i18n("查看模型信息(仅支持weights文件夹下提取的小模型文件)")
                )
                with gr.Row():
                    ckpt_path1 = gr.Textbox(
                        label=i18n("模型路径"), value="", interactive=True
                    )
                    but8 = gr.Button(i18n("查看"), variant="primary")
                    info6 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                but8.click(show_info, [ckpt_path1], info6, api_name="ckpt_show")
            with gr.Group():
                gr.Markdown(
                    value=i18n(
                        "模型提取(输入logs文件夹下大文件模型路径),适用于训一半不想训了模型没有自动提取保存小文件模型,或者想测试中间模型的情况"
                    )
                )
                with gr.Row():
                    ckpt_path2 = gr.Textbox(
                        label=i18n("模型路径"),
                        value="E:\\codes\\py39\\logs\\mi-test_f0_48k\\G_23333.pth",
                        interactive=True,
                    )
                    save_name = gr.Textbox(
                        label=i18n("保存名"), value="", interactive=True
                    )
                    sr__ = gr.Radio(
                        label=i18n("目标采样率"),
                        choices=["32k", "40k", "48k"],
                        value="40k",
                        interactive=True,
                    )
                    if_f0__ = gr.Radio(
                        label=i18n("模型是否带音高指导,1是0否"),
                        choices=["1", "0"],
                        value="1",
                        interactive=True,
                    )
                    version_1 = gr.Radio(
                        label=i18n("模型版本型号"),
                        choices=["v1", "v2"],
                        value="v2",
                        interactive=True,
                    )
                    info___ = gr.Textbox(
                        label=i18n("要置入的模型信息"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    but9 = gr.Button(i18n("提取"), variant="primary")
                    info7 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                    ckpt_path2.change(
                        change_info_, [ckpt_path2], [sr__, if_f0__, version_1]
                    )
                but9.click(
                    extract_small_model,
                    [ckpt_path2, save_name, sr__, if_f0__, info___, version_1],
                    info7,
                    api_name="ckpt_extract",
                )

        with gr.TabItem(i18n("Onnx导出")):
            with gr.Row():
                ckpt_dir = gr.Textbox(
                    label=i18n("RVC模型路径"), value="", interactive=True
                )
            with gr.Row():
                onnx_dir = gr.Textbox(
                    label=i18n("Onnx输出路径"), value="", interactive=True
                )
            with gr.Row():
                infoOnnx = gr.Label(label="info")
            with gr.Row():
                butOnnx = gr.Button(i18n("导出Onnx模型"), variant="primary")
            butOnnx.click(
                export_onnx, [ckpt_dir, onnx_dir], infoOnnx, api_name="export_onnx"
            )

        tab_faq = i18n("常见问题解答")
        with gr.TabItem(tab_faq):
            try:
                if tab_faq == "常见问题解答":
                    with open("docs/cn/faq.md", "r", encoding="utf8") as f:
                        info = f.read()
                else:
                    with open("docs/en/faq_en.md", "r", encoding="utf8") as f:
                        info = f.read()
                gr.Markdown(value=info)
            except:
                gr.Markdown(traceback.format_exc())

    if config.iscolab:
        logger.info("Launching Gradio (colab mode)")
        # 511 并发在本地/非GPU环境下容易触发底层库不稳定（甚至崩溃）
        app.queue(concurrency_count=32, max_size=1022).launch(
            share=True,
            show_error=True,
            debug=True,
        )
    else:
        logger.info("Launching Gradio on 0.0.0.0:%s", config.listen_port)
        app.queue(concurrency_count=32, max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=False,
            show_error=True,
            debug=True,
        )
