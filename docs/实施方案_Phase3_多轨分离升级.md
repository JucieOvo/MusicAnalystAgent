# 实施方案：Phase 3 - 听觉分离专家 (Separator) 多轨升级

## 1. 背景与目标
当前系统使用的 `BS-RoFormer` 模型配置为单轨输出 (`num_stems: 1`)，仅能分离人声 (Vocals) 和伴奏 (Instrumental)。
用户要求“将歌曲中的每一种乐器分为独立的音轨”，即实现更细粒度的分离（如 Vocals, Drums, Bass, Piano, Guitar, Other）。

## 2. 技术选型
为了满足“分离每一种乐器”的高级需求，单纯使用 BS-RoFormer（通常为 4 stems）可能不够。
我们建议引入 **Hybrid Transformer Demucs (HTDemucs)** 模型，特别是其 6-stem 变体 (`htdemucs_6s`)，它可以分离以下轨道：
*   **Vocals** (人声)
*   **Drums** (鼓)
*   **Bass** (贝斯)
*   **Guitar** (吉他)
*   **Piano** (钢琴)
*   **Other** (其他)

为了简化集成并获得最佳性能，我们将使用 **`audio-separator`** 库。这是一个封装了 Demucs, MDX-Net, VR Arch 等 SOTA 模型的工业级库，支持 GPU 加速和模型自动管理。

## 3. 架构变更

### 3.1 依赖更新
移除对手写 `bs_roformer` 推理代码的依赖（或将其作为遗留选项），引入 `audio-separator`。
```bash
pip install "audio-separator[gpu]"
```

### 3.2 配置变更 (`src/config.py`)
新增 `AudioSeparatorConfig`，支持选择模型类型（如 `htdemucs_6s`）。

```python
class SeparatorConfig(BaseModel):
    model_name: str = Field(default="htdemucs_6s", description="分离模型名称")
    output_format: str = Field(default="wav", description="输出格式")
    use_gpu: bool = Field(default=True, description="是否使用 GPU")
```

### 3.3 代码重构 (`src/agents/separator.py`)
重写 `AudioSeparator` 类：
1.  **初始化**: 实例化 `audio_separator.separator.Separator`。
2.  **模型加载**: 自动下载并加载 `htdemucs_6s` 权重。
3.  **推理**: 调用 `separate()` 方法。
4.  **输出处理**: 将生成的 6 个文件重命名并移动到规范的 `stems` 目录。

## 4. 预期效果
*   **输入**: 单个音频文件 (如 `test.mp3`)
*   **输出**: 文件夹 `output/test/stems/` 包含：
    *   `vocals.wav`
    *   `drums.wav`
    *   `bass.wav`
    *   `guitar.wav`
    *   `piano.wav`
    *   `other.wav`

## 5. 实施步骤
1.  **安装依赖**: 安装 `audio-separator`。
2.  **修改配置**: 更新 `src/config.py`。
3.  **重构代码**: 重写 `src/agents/separator.py`。
4.  **验证测试**: 使用 `卡农.mp3` 进行测试，检查是否生成了钢琴轨道（卡农主要是钢琴/弦乐，预期会有 Piano 和 Other/Strings）。

## 6. 风险控制
*   **显存占用**: Demucs 6s 显存占用适中，但在 4GB 显存以下显卡可能需要使用 `segment_size` 分块处理。
*   **依赖冲突**: `audio-separator` 依赖较新的 `onnxruntime-gpu` 和 `torch`，需确保环境兼容。
