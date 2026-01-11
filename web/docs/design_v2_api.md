# RVC WebUI v2 - API 接口设计

## 1. 基础规范

*   **Prefix**: `/api/v2`
*   **Format**: JSON
*   **Errors**: 标准 HTTP 状态码 + `{ "detail": "error message" }`

## 2. 离线推理 (Inference)

### 2.1 任务管理
*   `POST /inference/tasks`
    *   创建新的推理任务。
    *   Body: `{ model_id, audio_path, params: { ... } }`
*   `GET /inference/tasks`
    *   获取推理任务列表 (分页)。
    *   Query: `limit=20, offset=0`
*   `GET /inference/tasks/{id}`
    *   获取单个任务详情与状态。
*   `POST /inference/tasks/{id}/cancel`
    *   取消正在排队或运行的任务。
*   `DELETE /inference/tasks/{id}`
    *   删除任务记录及产物。

### 2.2 资源
*   `GET /models`
    *   列出所有可用模型 (扫描 `assets/weights` + 数据库元数据)。
*   `GET /indexes`
    *   列出所有可用索引文件。

---

## 3. 模型训练 (Model Training)

### 3.1 训练项目管理
*   `POST /trainings`
    *   新建训练项目。
    *   Body: `{ name, target_sr, if_f0, version, dataset_path }`
*   `GET /trainings`
    *   获取训练项目列表。
*   `GET /trainings/{id}`
    *   获取特定训练的完整状态（当前步骤、配置、路径）。
*   `PUT /trainings/{id}`
    *   更新配置或状态。
*   `DELETE /trainings/{id}`
    *   删除训练项目（可选：是否同时删除磁盘文件）。

### 3.2 流程控制 (Steps)

#### Step 1: 预处理 (Preprocess)
*   `POST /trainings/{id}/preprocess`
    *   触发预处理任务。
    *   Body: `{ thread_count: 8, ... }`
*   `GET /trainings/{id}/preprocess/samples`
    *   获取预处理后的随机音频切片列表（用于前端试听验证）。

#### Step 2: 特征提取 (Extract)
*   `POST /trainings/{id}/extract`
    *   触发特征提取任务。
    *   Body: `{ f0_method: "rmvpe", gpu_id: 0, ... }`

#### Step 3: 训练 (Train)
*   `POST /trainings/{id}/train/start`
    *   启动或恢复训练进程。
    *   Body: `{ total_epoch: 100, batch_size: 8, save_every_epoch: 5, ... }`
*   `POST /trainings/{id}/train/pause`
    *   暂停训练（保存 ckpt 后停止进程）。
*   `GET /trainings/{id}/train/metrics`
    *   获取训练曲线数据 (Loss, LR)。
    *   Query: `since_step=100` (用于增量更新图表)。

#### Step 4: 索引与导出 (Index & Export)
*   `POST /trainings/{id}/index`
    *   训练 Faiss 索引。
*   `POST /trainings/{id}/export`
    *   导出最终模型。
    *   Body: `{ checkpoint_name: "G_1000.pth", export_name: "MyModel_Final" }`

---

## 4. 系统 (System)

*   `GET /system/info`
    *   获取 GPU 列表、显存状态、CPU 核心数。
*   `POST /system/upload`
    *   通用文件上传接口（用于上传数据集、音频）。
    *   Return: `{ path: "/absolute/path/to/file" }`
*   `GET /system/files/browse`
    *   简单的文件浏览器接口（用于选择服务器上的数据集路径）。
    *   Query: `path=/Users`

## 5. WebSocket 实时流

*   `WS /ws/tasks/{task_id}/log`
    *   订阅特定任务的实时日志输出。
*   `WS /ws/trainings/{id}/log`
    *   订阅训练主进程的实时日志。
