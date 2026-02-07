# BS-RoFormer 验证与 CLaMP 3 集成实施方案

> **编写日期**：2026-02-07
> **摘要**：本方案旨在验证 BS-RoFormer 音源分离（Phase 1）的完成度，并详细规划 CLaMP 3 语义分析模型（Phase 2）的集成工作。

---

## 一、当前项目状态回顾

根据最新代码审查：

1.  **Phase 1: BS-RoFormer 音源分离**
    -   **状态**：代码逻辑主体 (`separator.py`) 已实现，包含模型加载、分块推理及多 Stem 输出逻辑。
    -   **缺失**：尚未进行环境依赖验证和实际运行测试，依赖库 `BS-RoFormer` 需确认安装。
    -   **风险**：模型权重文件完整性及显存占用情况需实际测试。

2.  **Phase 2: CLaMP 3 语义分析**
    -   **状态**：处于模拟阶段 (`semantic_reviewer.py`)。
    -   **缺失**：核心模型加载、音频编码、文本检索逻辑均为 Mock 实现。
    -   **任务**：需真正集成 CLaMP 3 模型，替换硬编码逻辑。

---

## 二、Phase 1: BS-RoFormer 验证计划

在推进新功能前，必须确保基础音源分离功能正常运作。

### 2.1 依赖环境检查
-   **目标**：确认 `requirements.txt` 中的依赖已安装，特别是 `BS-RoFormer`。
-   **操作**：运行 `pip install -r requirements.txt` (使用清华源)。

### 2.2 模型权重检查
-   **目标**：验证模型文件完整性。
-   **检查项**：
    -   `models/model_bs_roformer_ep_317_sdr_12.9755.ckpt` (应约为 639MB)
    -   `models/model_bs_roformer_ep_317_sdr_12.9755.yaml`

### 2.3 功能测试
-   **目标**：跑通完整的音频分离流程。
-   **测试脚本**：
    ```bash
    python -m src.agents.separator data/test_audio.mp3
    ```
-   **验证点**：
    1.  模型加载是否成功（显存占用正常）。
    2.  长音频分块推理是否无报错。
    3.  输出文件 `vocals.wav` 和 `instrumental.wav` 是否生成且听感正常。

---

## 三、Phase 2: CLaMP 3 集成实施方案

本阶段将使 `semantic_reviewer.py` 具备真实的语义理解能力。

### 3.1 依赖安装
-   新增依赖：`transformers`, `torch`, `torchaudio`
-   确认安装命令：
    ```bash
    pip install transformers torch torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

### 3.2 核心模块开发 (`semantic_reviewer.py`)

#### 3.2.1 模型加载 (`SemanticAnalyzer.load_model`)
-   **逻辑**：
    1.  使用 `transformers` 库加载预训练的 CLaMP 3 模型。
    2.  检测本地缓存，若无则自动下载（约 1GB）。
    3.  配置设备（CUDA/CPU）。
-   **代码变更**：
    -   移除 try-except pass 块。
    -   实现 `AutoModel.from_pretrained("sander-wood/clamp-small-500m")` (暂定使用 Small 版本以平衡性能)。
    -   实现 `AutoProcessor`。

#### 3.2.2 描述符库处理 (`DescriptorBank`)
-   **逻辑**：
    1.  遍历 `DEFAULT_DESCRIPTORS` 中的所有标签。
    2.  调用 CLaMP 3 文本编码器生成 Text Embeddings。
    3.  将 Embeddings 缓存为 `.npy` 文件，避免重复计算。
-   **代码变更**：
    -   实现 `compute_embeddings(model, processor)` 方法。
    -   实现 `load_cached_embeddings()` 方法。

#### 3.2.3 音频编码与检索 (`SemanticAnalyzer`)
-   **逻辑**：
    1.  **音频预处理**：加载音频，重采样至 48kHz，截取片段（如中间 30 秒）。
    2.  **音频编码**：调用 CLaMP 3 音频编码器生成 Audio Embedding。
    3.  **相似度计算**：计算 Audio Embedding 与所有 Text Embeddings 的余弦相似度。
    4.  **Top-K 筛选**：按类别（Mood, Genre 等）筛选高分标签。
-   **代码变更**：
    -   实现 `encode_audio`。
    -   实现 `retrieve_tags` 的真实计算逻辑。

### 3.3 集成测试
-   **测试用例**：使用已知风格的音频（如《卡农》）进行测试。
-   **预期结果**：
    -   Genre 应包含 "Classical"。
    -   Instruments 应包含 "Piano" 或 "Strings"。
    -   Mood 应包含 "Peaceful" 或 "Calm"。

---

## 四、执行请求

请批准以下操作：
1.  **执行 Phase 1 验证**：检查并安装依赖，运行 `separator.py` 测试。
2.  **执行 Phase 2 开发**：修改 `semantic_reviewer.py`，实现真实的 CLaMP 3 集成。

> **注意**：所有代码修改将严格遵循中文注释规范，并保持模块化设计。
