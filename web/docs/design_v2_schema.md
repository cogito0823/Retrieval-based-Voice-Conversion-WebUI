# RVC WebUI v2 - 数据库设计 (Schema)

## 1. 设计原则

*   **轻量级**: 使用 SQLite，单文件存储，无需额外部署数据库服务。
*   **ORM**: 建议引入 `SQLAlchemy` (Python) 或 `Prisma` (如果前端直连，但在 FastAPI 架构下推荐 SQLAlchemy + Pydantic)。考虑到项目依赖最小化，我们将使用 Python 标准库 `sqlite3` 或轻量级封装，但设计上遵循关系型规范。
*   **状态管理**: 核心状态（如训练步骤、任务进度）必须持久化，防止服务重启后丢失。

## 2. 表结构定义 (ER Diagram)

### 2.1 Table: `trainings` (模型训练项目)

存储一个模型训练的全生命周期数据。

| 字段名 | 类型 | 必填 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **id** | TEXT | Yes | (UUID) | 主键 |
| **name** | TEXT | Yes | - | 模型名称 (唯一) |
| **status** | TEXT | Yes | `draft` | 状态: `draft` (草稿), `ready` (就绪), `training` (训练中), `finished` (完成), `failed` (失败) |
| **created_at** | INTEGER | Yes | (Now) | 创建时间戳 (ms) |
| **updated_at** | INTEGER | Yes | (Now) | 更新时间戳 (ms) |
| **dataset_path** | TEXT | No | - | 原始数据集路径 (Zip 或 目录) |
| **target_sr** | INTEGER | Yes | 40000 | 目标采样率: 40000, 48000 |
| **has_preprocessed** | BOOLEAN | Yes | FALSE | 步骤1：数据预处理是否完成 |
| **has_extracted** | BOOLEAN | Yes | FALSE | 步骤2：特征提取是否完成 |
| **f0_method** | TEXT | No | `rmvpe` | 提取特征使用的 F0 算法 |
| **model_version** | TEXT | Yes | `v2` | 模型版本: `v1`, `v2` |
| **gpu_id** | TEXT | No | `0` | 绑定的 GPU ID (多卡环境预留) |
| **result_model_path** | TEXT | No | - | 最终导出的模型路径 |

### 2.2 Table: `training_tasks` (训练子任务历史)

记录一次训练项目中执行过的所有耗时操作（预处理、提取、训练）。
一对多关系：`trainings.id` -> `training_tasks.training_id`

| 字段名 | 类型 | 必填 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **id** | TEXT | Yes | (UUID) | 主键 (Task ID) |
| **training_id** | TEXT | Yes | - | 外键 -> trainings.id |
| **step_type** | TEXT | Yes | - | `preprocess`, `extract_f0`, `extract_feature`, `train_model`, `train_index` |
| **status** | TEXT | Yes | `queued` | `queued`, `running`, `success`, `failed`, `cancelled` |
| **pid** | INTEGER | No | - | 操作系统进程 ID (用于终止) |
| **config_snapshot** | JSON | No | - | 执行时的参数快照 (e.g. {batch_size: 8, epochs: 100}) |
| **error_msg** | TEXT | No | - | 失败时的错误信息 |
| **created_at** | INTEGER | Yes | (Now) | - |
| **finished_at** | INTEGER | No | - | - |

### 2.3 Table: `inference_tasks` (离线推理任务)

记录离线推理的历史。

| 字段名 | 类型 | 必填 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **id** | TEXT | Yes | (UUID) | 主键 |
| **model_id** | TEXT | Yes | - | 使用的模型 ID (文件名) |
| **status** | TEXT | Yes | `queued` | `queued`, `running`, `success`, `failed`, `cancelled` |
| **input_audio_path** | TEXT | Yes | - | 输入音频路径 |
| **output_audio_path** | TEXT | No | - | 输出音频路径 |
| **params** | JSON | Yes | - | 推理参数 (f0_method, pitch, index_rate...) |
| **created_at** | INTEGER | Yes | (Now) | - |
| **cost_ms** | INTEGER | No | - | 耗时 (毫秒) |

### 2.4 Table: `system_logs` (可选)

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **id** | INTEGER | 自增主键 |
| **level** | TEXT | INFO, WARN, ERROR |
| **module** | TEXT | `trainer`, `inference`, `system` |
| **message** | TEXT | 日志内容 |
| **created_at** | INTEGER | - |

## 3. 目录与文件映射

除了数据库，文件系统结构也需规范化，以配合 `trainings` 表。

```
logs/
  {training_name}/         # 对应 trainings.name
    config.json            # 训练配置
    train_events.jsonl     # 训练过程详细日志 (WebUI 专用)
    0_gt_wavs/             # 预处理产物
    1_16k_wavs/            # 预处理产物
    2a_f0/                 # 特征提取产物
    2b-f0nsf/              # 特征提取产物
    3_feature256/          # 特征提取产物
    G_*.pth                # Checkpoints
    D_*.pth                # Checkpoints
    added_*.index          # 索引文件
```

WebUI 后端将通过读取数据库获取状态，通过读取 `logs/{name}/train_events.jsonl` 获取实时训练曲线数据。
