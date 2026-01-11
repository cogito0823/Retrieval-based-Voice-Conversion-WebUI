# RVC WebUI 重构概要设计 (v2)

## 1. 背景与目标

本设计旨在重构 RVC WebUI，从单一的脚本工具升级为功能完整、体验流畅的 Web 应用程序。
核心目标是**统一离线推理与模型训练的工作流**，降低用户使用门槛，并提供更强的任务管理与状态监控能力。

## 2. 核心概念

### 2.1 训练 (Training)
训练流程将围绕 **“模型训练 (Model Training)”** 这一核心实体进行组织（原名为“实验”）。
*   一个训练对应一个模型的全生命周期。
*   训练包含：配置参数、数据集引用、预处理中间产物、训练日志、Checkpoints。
*   训练目录结构化，用户无需手动去 `logs/` 下寻找文件。

### 2.2 任务队列 (Task Queue)
所有的耗时操作（推理、预处理、训练）均封装为异步任务。
*   支持排队、取消、进度实时推送。
*   前端不再阻塞等待，防止页面刷新导致任务丢失。
*   **双视图设计**：对于离线推理和模型训练，均提供“新建任务”向导和“任务列表”看板。

---

## 3. 功能清单 (Feature List)

### 3.1 核心模块 A：离线推理 (Inference)

面向最终用户，提供简单直观的语音转换能力。

| 功能点 | 说明 |
| :--- | :--- |
| **新建推理任务** | 独立的创建页面/弹窗，引导用户完成参数配置。 |
| **推理任务列表** | 独立的列表页面，展示所有历史推理任务的状态（排队/进行中/成功/失败）、耗时、参数快照。 |
| **模型选择器** | 可视化选择本地模型，显示模型封面、名称、训练步数信息。 |
| **音频上传** | 支持拖拽上传单曲或批量上传文件夹（自动识别音频文件）。 |
| **高级推理参数** | • F0 预测算法 (rmvpe, fcpe, harvest, pm)<br>• 变调 (Pitch/Key)<br>• 索引率 (Index Rate)<br>• 混合率 (Volume Mix)<br>• 保护阈值 (Protect)<br>• *新增：F0 曲线文件导入 (可选)* |
| **结果管理** | 任务详情页直接试听结果，支持波形图预览；提供单个下载或批量 Zip 打包下载。 |

### 3.2 核心模块 B：模型训练 (Model Training)

面向进阶用户，提供从数据处理到模型导出的全流程向导。

#### B.1 模型训练管理 (Model Training Management)
*   **新建训练**：
    *   设置模型名称 (Model Name)。
    *   选择目标采样率 (40k/48k)。
    *   选择版本 (v2)。
    *   选择硬件设备 (GPU/CPU)。
*   **训练列表**：查看所有训练项目的状态（草稿/进行中/已完成），支持从列表快速进入详情或继续训练。

#### B.2 数据准备 (Step 1: Data Preprocessing)
*   **数据集导入**：
    *   支持上传 Zip 压缩包（自动解压）。
    *   支持填写服务器本地绝对路径。
*   **音频预处理** (异步任务)：
    *   调用 `infer/modules/train/preprocess.py`。
    *   **参数**：SR (采样率), N_Process (线程数)。
    *   **产物**：生成 `1_16k_wavs` (16k采样率目录) 和 `2a_f0` 等中间文件。
    *   **结果确认**：显示切片后的总片段数，允许随机试听几个切片。

#### B.3 特征提取 (Step 2: Feature Extraction)
*   **配置**：
    *   选择 F0 提取算法（rmvpe/crepe/pm/harvest）。
    *   指定 GPU 卡号。
*   **执行** (异步任务)：
    *   调用 `infer/modules/train/extract/extract_f0_print.py` (提取 Pitch)。
    *   调用 `infer/modules/train/extract/extract_feature_print.py` (提取 Embedding)。
*   **监控**：显示提取进度条 (e.g., "Extracting f0: 100/500 files")。

#### B.4 模型训练 (Step 3: Model Training)
*   **训练配置**：
    *   Batch Size (批量大小)。
    *   Total Epochs (总轮数)。
    *   Save Every Epoch (保存间隔)。
    *   FP16 Run (是否开启半精度)。
    *   Cache Data in GPU (显存缓存)。
    *   Pretrained Models (底模路径 G/D)。
*   **训练控制**：
    *   调用 `infer/modules/train/train.py`。
    *   **启动/继续**：前端通过 API 触发训练进程。
    *   **暂停**：发送信号优雅停止（保存当前状态）。
    *   **终止**：强制结束进程。
*   **可视化监控**：
    *   解析训练日志流，前端实时绘制 Loss (G/D/Mel/KL) 曲线。
    *   显示当前 Epoch、Step、LR、剩余时间预估。

#### B.5 索引与导出 (Step 4: Index & Export)
*   **训练索引 (Index)**：
    *   调用 `infer/modules/train/train_index.py` (需确认具体脚本名，通常是 faiss 训练)。
    *   生成 `.index` 文件，用于推理时增强音色还原度。
*   **模型融合 (Export)**：
    *   选择特定的 Checkpoint (e.g., `G_23333.pth`)。
    *   提取权重，导出最终的推理模型文件。
    *   **发布**：将模型+索引一键发布到系统的“模型库”，使其在推理页面可见。

### 3.3 系统管理 (System)

| 功能点 | 说明 |
| :--- | :--- |
| **硬件监控** | 顶部栏常驻显示 GPU 显存使用率、温度、利用率。 |
| **设置** | 全局路径配置、语言切换 (i18n)。 |

---

## 4. 数据模型设计 (Schema Draft)

为了支撑上述功能，引入 SQLite 数据库。

### 4.1 Table: tasks (通用任务表)
| Field | Type | Note |
| :--- | :--- | :--- |
| id | TEXT | UUID, Primary Key |
| type | TEXT | `infer_offline`, `train_preprocess`, `train_extract`, `train_model` |
| status | TEXT | `queued`, `running`, `success`, `failed`, `cancelled` |
| params | JSON | 任务参数快照 |
| created_at | INTEGER | Timestamp |
| result | JSON | 任务结果（如 artifactId, metrics） |
| error | TEXT | 错误信息 |
| related_id | TEXT | 关联的 training_id (若是训练任务) |

### 4.2 Table: trainings (训练项目表)
| Field | Type | Note |
| :--- | :--- | :--- |
| id | TEXT | UUID, Primary Key |
| name | TEXT | 模型名称 |
| status | TEXT | `draft`, `processing`, `training`, `finished` |
| config | JSON | 训练配置 (lr, batch_size, epochs...) |
| dataset_path | TEXT | 数据集路径 |
| current_step | INTEGER | 当前进行到的步骤 (1-4) |
| created_at | INTEGER | Timestamp |

---

## 5. 架构简述

*   **前端**: React + Vite + TailwindCSS (SPA)。
*   **后端**: Python FastAPI。
*   **中间件**: 
    *   **SQLite**: 持久化任务和训练元数据。
    *   **TaskManager**: 内存中的任务队列管理器，负责进程调度和日志流转发。
    *   **WebSocket**: `/ws/tasks/{id}/log` 用于实时推送任务日志。
