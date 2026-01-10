import traceback
import logging

logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf
import torch
from io import BytesIO
import time
import uuid
import os
import queue
import threading
from time import monotonic

from infer.lib.audio import load_audio, wav2
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.modules.vc.pipeline import Pipeline
from infer.modules.vc.utils import *


def _silent_audio(sr: int = 16000):
    """
    Gradio 的 Audio 组件不接受 (None, None)；
    返回一个极短的静音片段用于占位，避免前端/后处理报错。
    """
    return _audio_to_gradio(sr, np.zeros(1600, dtype=np.int16))


def _audio_to_gradio(sr: int, audio: np.ndarray):
    """
    某些环境下直接返回 (sr, np.ndarray) 会导致前端显示 Error 但后端无 traceback（数据过大/序列化问题）。
    这里统一写到临时 wav 文件并返回路径，交给 gr.Audio 读取。
    """
    try:
        tmp_dir = os.environ.get("TEMP") or os.path.join(os.getcwd(), "TEMP")
        os.makedirs(tmp_dir, exist_ok=True)
        out_path = os.path.join(
            tmp_dir, f"rvc_vc_{int(time.time())}_{uuid.uuid4().hex}.wav"
        )
        # Pipeline 输出通常是 int16；明确写成 PCM_16
        sf.write(out_path, audio, sr, subtype="PCM_16")
        return out_path
    except Exception:
        # 回退：仍返回 tuple，至少不让函数崩
        return (sr, audio)


def _fmt_secs(s: float) -> str:
    s = max(0, int(s))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{ss:02d}"
    return f"{m:02d}:{ss:02d}"


class VC:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.version = None
        self.hubert_model = None

        self.config = config

    def get_vc(self, sid, *to_return_protect):
        logger.info("Get sid: " + sid)

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5
            ),
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33
            ),
            "__type__": "update",
        }

        if sid == "" or sid == []:
            if (
                self.hubert_model is not None
            ):  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
                logger.info("Clean model cache")
                del (self.net_g, self.n_spk, self.hubert_model, self.tgt_sr)  # ,cpt
                self.hubert_model = self.net_g = self.n_spk = self.hubert_model = (
                    self.tgt_sr
                ) = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ###楼下不这么折腾清理不干净
                self.if_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", "v1")
                if self.version == "v1":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs256NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.cpt["config"])
                elif self.version == "v2":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs768NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
                del self.net_g, self.cpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return (
                {"visible": False, "__type__": "update"},
                {
                    "visible": True,
                    "value": to_return_protect0,
                    "__type__": "update",
                },
                {
                    "visible": True,
                    "value": to_return_protect1,
                    "__type__": "update",
                },
                "",
                "",
            )
        person = f'{os.getenv("weight_root")}/{sid}'
        logger.info(f"Loading: {person}")

        self.cpt = torch.load(person, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        n_spk = self.cpt["config"][-3]
        index = {"value": get_index_path_from_model(sid), "__type__": "update"}
        logger.info("Select index: " + index["value"])

        return (
            (
                {"visible": True, "maximum": n_spk, "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                index,
                index,
            )
            if to_return_protect
            else {"visible": True, "maximum": n_spk, "__type__": "update"}
        )

    def vc_single(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        # 用后台线程执行推理，generator 持续回传进度，避免前端超时显示 Error
        q: "queue.Queue[tuple[str, object]]" = queue.Queue()
        silent = _silent_audio()

        def emit(msg: str):
            try:
                q.put(("msg", msg))
            except Exception:
                pass

        def done(text: str, audio_out):
            try:
                q.put(("done", (text, audio_out)))
            except Exception:
                pass

        def worker():
            try:
                t0 = monotonic()
                # 进度权重（总和 1.0）
                weights = {
                    "audio": 0.06,   # 读取音频
                    "hubert": 0.14,  # 加载 HuBERT（首次）
                    "index": 0.06,   # 加载索引/缓存
                    "f0": 0.24,      # 提取 F0（含 rmvpe 初始化）
                    "infer": 0.40,   # 分段推理
                    "post": 0.10,    # 后处理与写文件
                }
                stage_frac = {k: 0.0 for k in weights.keys()}

                def overall_pct() -> float:
                    return 100.0 * sum(weights[k] * stage_frac.get(k, 0.0) for k in weights)

                def emit_progress(desc: str, extra: str = ""):
                    pct = overall_pct()
                    elapsed = monotonic() - t0
                    if pct > 0.5:
                        eta = elapsed * (100.0 - pct) / pct
                        eta_s = _fmt_secs(eta)
                    else:
                        eta_s = "--:--"
                    msg = f"{desc}  {pct:5.1f}% | 已用 {_fmt_secs(elapsed)} | 预计剩余 {eta_s}"
                    if extra:
                        msg += f"\n{extra}"
                    emit(msg)

                logger.info(
                    "vc_single called: sid=%s, input_audio_path=%r, f0_method=%s, file_index=%r, file_index2=%r",
                    sid,
                    input_audio_path,
                    f0_method,
                    file_index,
                    file_index2,
                )
                emit_progress("开始推理…")

                if input_audio_path is None or input_audio_path == "":
                    done("请填写输入音频路径（或上传音频）。", silent)
                    return

                if self.pipeline is None or self.net_g is None:
                    done("请先在【推理音色】下拉框选择模型并等待加载完成，然后再点击【转换】。", silent)
                    return

                try:
                    f0_up_key_i = int(f0_up_key)
                except Exception:
                    f0_up_key_i = 0

                emit_progress("正在读取输入音频…")
                audio = load_audio(input_audio_path, 16000)
                stage_frac["audio"] = 1.0
                audio_max = np.abs(audio).max() / 0.95
                if audio_max > 1:
                    audio /= audio_max
                times = [0, 0, 0]

                if self.hubert_model is None:
                    emit_progress("正在加载 HuBERT…（首次会较慢）")
                    self.hubert_model = load_hubert(self.config)
                stage_frac["hubert"] = 1.0

                if file_index:
                    file_index_ = (
                        file_index.strip(" ")
                        .strip('"')
                        .strip("\n")
                        .strip('"')
                        .strip(" ")
                        .replace("trained", "added")
                    )
                elif file_index2:
                    file_index_ = file_index2
                else:
                    file_index_ = ""

                def progress_cb(*args):
                    # 兼容：
                    # - (cur,total,desc)
                    # - (stage,cur,total,desc)
                    stage = "infer"
                    cur = 0
                    total = 0
                    desc = "处理中…"
                    if len(args) == 3:
                        cur, total, desc = args
                    elif len(args) >= 4:
                        stage, cur, total, desc = args[:4]

                    if stage in stage_frac:
                        if total and total > 0:
                            stage_frac[stage] = min(1.0, max(0.0, float(cur) / float(total)))
                            extra = f"{desc}（{cur}/{total}）"
                        else:
                            extra = str(desc)
                            if "完成" in str(desc) or "就绪" in str(desc):
                                stage_frac[stage] = 1.0
                            else:
                                stage_frac[stage] = max(stage_frac[stage], 0.01)
                        emit_progress(str(desc), extra=extra)
                    else:
                        emit_progress(str(desc), extra=f"{desc}（{cur}/{total}）" if total else "")

                emit_progress("准备进入推理…")
                audio_opt = self.pipeline.pipeline(
                    self.hubert_model,
                    self.net_g,
                    sid,
                    audio,
                    input_audio_path,
                    times,
                    f0_up_key_i,
                    f0_method,
                    file_index_,
                    index_rate,
                    self.if_f0,
                    filter_radius,
                    self.tgt_sr,
                    resample_sr,
                    rms_mix_rate,
                    self.version,
                    protect,
                    f0_file,
                    progress_cb=progress_cb,
                )
                stage_frac["infer"] = 1.0
                stage_frac["post"] = 1.0

                if self.tgt_sr != resample_sr >= 16000:
                    tgt_sr = resample_sr
                else:
                    tgt_sr = self.tgt_sr

                index_info = (
                    "Index:\n%s." % file_index_
                    if file_index_ and os.path.exists(file_index_)
                    else "Index not used."
                )
                out_path = _audio_to_gradio(tgt_sr, audio_opt)
                done(
                    "Success.\n%s\nTime:\nnpy: %.2fs, f0: %.2fs, infer: %.2fs."
                    % (index_info, *times),
                    out_path,
                )
            except Exception as e:
                info = traceback.format_exc()
                logger.warning(info)
                done(f"推理失败：{e}\n\n{info}", silent)

        threading.Thread(target=worker, daemon=True).start()

        # 主循环：持续输出进度
        last_msg = "推理中…"
        last_audio = silent
        while True:
            try:
                kind, payload = q.get(timeout=1.0)
            except queue.Empty:
                # heartbeat，避免前端认为无响应
                yield last_msg, last_audio
                continue
            if kind == "msg":
                last_msg = str(payload)
                yield last_msg, last_audio
            elif kind == "done":
                text, audio_out = payload
                yield str(text), audio_out
                break

    def vc_multi(
        self,
        sid,
        dir_path,
        opt_root,
        paths,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        format1,
    ):
        try:
            dir_path = (
                dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )  # 防止小白拷路径头尾带了空格和"和回车
            opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            os.makedirs(opt_root, exist_ok=True)
            try:
                if dir_path != "":
                    paths = [
                        os.path.join(dir_path, name) for name in os.listdir(dir_path)
                    ]
                else:
                    paths = [path.name for path in paths]
            except:
                traceback.print_exc()
                paths = [path.name for path in paths]
            infos = []
            for path in paths:
                info, opt = self.vc_single(
                    sid,
                    path,
                    f0_up_key,
                    None,
                    f0_method,
                    file_index,
                    file_index2,
                    # file_big_npy,
                    index_rate,
                    filter_radius,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                )
                if "Success" in info:
                    try:
                        tgt_sr, audio_opt = opt
                        if format1 in ["wav", "flac"]:
                            sf.write(
                                "%s/%s.%s"
                                % (opt_root, os.path.basename(path), format1),
                                audio_opt,
                                tgt_sr,
                            )
                        else:
                            path = "%s/%s.%s" % (
                                opt_root,
                                os.path.basename(path),
                                format1,
                            )
                            with BytesIO() as wavf:
                                sf.write(wavf, audio_opt, tgt_sr, format="wav")
                                wavf.seek(0, 0)
                                with open(path, "wb") as outf:
                                    wav2(wavf, outf, format1)
                    except:
                        info += traceback.format_exc()
                infos.append("%s->%s" % (os.path.basename(path), info))
                yield "\n".join(infos)
            yield "\n".join(infos)
        except:
            yield traceback.format_exc()
