# 实施方案：Phase 2 - CLaMP 3 原生模型集成 (已完成)

## 1. 背景与目标
根据用户“不允许更换模型”的严格要求，本项目已放弃 Hugging Face 的自动加载方式，成功基于 `lib/clamp3` 目录下的原生代码实现了 `SemanticReviewer`。CLaMP 3 模型已成功加载，并正确执行音频与文本的语义对齐分析。

## 2. 技术架构分析

### 2.1 模型依赖链
CLaMP 3 的音频分析流程包含两个阶段：
1.  **音频特征提取 (Pre-encoder)**：
    *   模型：`m-a-p/MERT-v1-95M`
    *   输入：原始音频波形 (24kHz)
    *   处理：分段（5秒窗口，无重叠），提取所有层特征并取平均。
    *   输出：MERT 特征序列 `[NumSegments, 768]`。
2.  **语义对齐 (Encoder)**：
    *   模型：`sander-wood/clamp3` (SAAS Variant)
    *   输入：MERT 特征序列
    *   输出：全局语义向量 (Global Semantic Vector, 768维)。

### 2.2 关键文件与配置
*   **配置**：`lib/clamp3/code/config.py` (定义了权重路径和模型参数)。
*   **模型定义**：`lib/clamp3/code/utils.py` 中的 `CLaMP3Model` 类。
*   **MERT 提取**：`lib/clamp3/preprocessing/audio/extract_mert.py`。
*   **实现主体**：`src/agents/semantic_reviewer.py`

## 3. 集成方案

### 3.1 依赖引入
`SemanticReviewer` 已引入本地模块：
```python
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'lib', 'clamp3', 'code'))
sys.path.append(os.path.join(os.getcwd(), 'lib', 'clamp3', 'preprocessing', 'audio'))

from utils import CLaMP3Model
from hf_pretrains import HuBERTFeature
from config import CLAMP3_WEIGHTS_PATH, ...
```

### 3.2 模型加载策略
已实现自动权重检查与下载逻辑：
1.  **MERT 模型**：使用 `HuBERTFeature` 加载 `m-a-p/MERT-v1-95M`。
2.  **CLaMP 3 权重**：自动检查并下载 `weights_clamp3_saas...pth` 到 `lib/clamp3/code` 目录。

### 3.3 特征提取流程
`SemanticReviewer` 的特征提取方法已实现：
*   **`encode_audio(audio_path)`**: 实现了加载、重采样、分段、MERT特征提取、CLaMP 3推理的全流程。
*   **`encode_text_batch(text_list)`**: 实现了基于 XLM-R Tokenizer 和 CLaMP 3 Text Encoder 的批量文本编码。

### 3.4 相似度计算
使用余弦相似度计算音频嵌入与描述符库中所有文本嵌入的匹配度，并返回 Top-K 标签。

## 4. 验证结果
1.  **单元测试**：
    *   运行命令：`python -m src.agents.semantic_reviewer "F:\MusicAnalystAgent\卡农.mp3"`
    *   结果：成功输出 Mood, Genre, Instruments 等维度的语义标签及其置信度。
2.  **集成测试 (semantic_only)**：
    *   运行命令：`python -m src.main analyze "F:\MusicAnalystAgent\卡农.mp3" --task semantic_only`
    *   结果：Workflow 正确执行，路由到 Semantic 节点，成功生成并保存分析报告。

## 5. 后续计划
*   **全流程联调**：在环境具备的情况下，验证包含音源分离和转录的 `full_analysis` 模式。
*   **细粒度分析**：未来可扩展 `analyze` 方法，利用 `stems_paths` 对分离后的音轨进行独立的语义分析（如单独分析人声的情感）。
