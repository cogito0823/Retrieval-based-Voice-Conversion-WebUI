import torch
import sys

try:
    print("Loading model...")
    model_path = "assets/hubert/hubert_base.pt"
    # 尝试用 torch.load 加载
    checkpoint = torch.load(model_path, map_location="cpu")
    print("torch.load success!")
    print("Keys:", checkpoint.keys())
    
    # 尝试用 fairseq 接口加载（如果环境里有）
    import fairseq
    print("Import fairseq success. Loading via checkpoint_utils...")
    models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        suffix="",
    )
    print("Fairseq load success!")
    
except Exception as e:
    print("Error loading model:")
    print(e)
    import traceback
    traceback.print_exc()
