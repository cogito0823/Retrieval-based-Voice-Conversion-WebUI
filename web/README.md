# Web (重构代码统一目录)

v1 目标：**离线推理（串行、单用户）**。

## 后端：FastAPI

启动（建议在 `rvcv2` conda 环境中）：

```bash
conda activate rvcv2
python -m uvicorn web.server.app.main:app --host 127.0.0.1 --port 8000
```

接口：
- `GET /api/health`
- `GET /api/models`（扫描 `assets/weights/*.pth`）
- `GET /api/indexes`（扫描 `assets/indices/**/*.index`）
- `POST /api/uploads/audio`
- `POST /api/tasks/infer-offline`
- `GET /api/tasks/{taskId}`
- `GET /api/tasks/{taskId}/logs`
- `GET /api/artifacts/{artifactId}`

## 前端：Vite + React + TS

```bash
cd web/app
npm install
npm run dev
```

开发期默认通过 Vite proxy 转发 `/api` 到 `http://127.0.0.1:8000`。

