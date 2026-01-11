# RVCv2 Web Server (v1: 离线推理，串行)

## 目录约定
- 模型：仓库根目录 `assets/weights/*.pth`
- 索引：仓库根目录 `assets/indices/**/*.index`
- 服务端存储：`web/server/storage/`（上传、任务日志、推理输出）

## 运行
在已激活的 conda 环境中（例如 `rvcv2`）：

```bash
python -m uvicorn web.server.app.main:app --host 0.0.0.0 --port 8000 --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/api/health
```

## 可选环境变量
- `RVCV2_WEIGHT_DIR`：模型目录（默认 `assets/weights`）
- `RVCV2_INDEX_DIR`：索引目录（默认 `assets/indices`）
- `RVCV2_STORAGE_DIR`：服务端存储目录（默认 `web/server/storage`）

