import os

from fairseq import checkpoint_utils
import torch


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(config):
    # PyTorch 2.6+ 默认 torch.load(weights_only=True) 会拒绝反序列化 fairseq 的部分对象，
    # 导致 hubert_base.pt 加载失败。这里对“本地 hubert_base.pt”做兼容处理：
    # - 优先使用 torch.serialization.safe_globals allowlist fairseq Dictionary
    # - 回退：临时把 torch.load 默认 weights_only 设为 False（仅用于本次加载）
    hubert_path = "assets/hubert/hubert_base.pt"

    safe_loaded = False
    try:
        from fairseq.data.dictionary import Dictionary  # type: ignore

        if hasattr(torch, "serialization") and hasattr(torch.serialization, "safe_globals"):
            with torch.serialization.safe_globals([Dictionary]):
                models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
                    [hubert_path],
                    suffix="",
                )
                safe_loaded = True
    except Exception:
        safe_loaded = False

    if not safe_loaded:
        # Fallback: force weights_only=False if supported
        import inspect

        _orig_torch_load = torch.load

        def _torch_load_compat(*args, **kwargs):
            try:
                sig = inspect.signature(_orig_torch_load)
                if "weights_only" in sig.parameters and "weights_only" not in kwargs:
                    kwargs["weights_only"] = False
            except Exception:
                pass
            return _orig_torch_load(*args, **kwargs)

        torch.load = _torch_load_compat  # type: ignore[assignment]
        try:
            models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
                [hubert_path],
                suffix="",
            )
        finally:
            torch.load = _orig_torch_load  # type: ignore[assignment]

    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
