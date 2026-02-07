# Analyst 综合分析能力增强方案

## 1. 背景与目标
当前 `Analyst Node`（综合分析节点）生成的音乐分析报告主要依赖于基础的元数据（文件名、时长）和尚未实现的特征分析占位符。用户反馈分析结果“数据太少，不够全面，不像真人”。

本方案旨在通过深度挖掘 MIDI 转录数据，提取丰富的音乐特征，并优化 Prompt 策略，使 LLM 能够生成具备专业深度、细节丰富且拟人化的音乐分析报告。

## 2. 现状分析
- **特征缺失**: `AudioTranscriber.analyze_features` 方法目前为空实现 (TODO)，导致 BPM、调性、和弦等核心数据缺失。
- **Prompt 单薄**: 传递给 LLM 的 Prompt 仅包含文件名和空的占位符，缺乏具体的音乐内容描述。
- **数据利用率低**: 虽然 Basic Pitch 已经生成了音符事件 (Note Events)，但这些数据未被统计分析。

## 3. 技术方案

### 3.1 数据 Schema 扩展 (`src/schemas.py`)
扩展 `MusicalFeatures` 模型，增加分轨详细统计字段，以便传递给 LLM。

```python
class StemAnalysis(BaseModel):
    """分轨详细分析"""
    stem_type: str
    note_count: int
    note_density: float = Field(description="音符密度(个/秒)")
    pitch_range: tuple[int, int] = Field(description="音域范围 (min, max)")
    average_velocity: float = Field(description="平均力度")
    active_ratio: float = Field(description="活跃时间比例")
    description: str = Field(description="基于统计的简短描述")

class MusicalFeatures(BaseModel):
    # ... 原有字段 ...
    stem_analyses: Dict[str, StemAnalysis] = Field(default_factory=dict)
```

### 3.2 特征提取算法实现 (`src/agents/transcriber.py`)
在 `analyze_features` 中实现以下逻辑：

1.  **基础统计**:
    - 遍历每个分轨的 `notes`。
    - 计算音域 (Min/Max Pitch)。
    - 计算密度 (Notes / Duration)。
    - 计算活跃度 (Total Note Duration / Total Track Duration)。

2.  **BPM 估算 (若无音频特征)**:
    - 计算所有相邻音符的起始时间差 (IOI)。
    - 构建 IOI 直方图，寻找最常见的间隔。
    - 转换为 BPM (60 / IOI)。

3.  **调性检测 (Key Detection)**:
    - Krumhansl-Schmuckler Key-Finding Algorithm 的简化版。
    - 统计全曲 Pitch Class (0-11) 加权分布（时长 x 力度）。
    - 与 Major/Minor Profile 进行相关性匹配，选出最佳 Key。

4.  **和弦进行推断 (Simple Chord Inference)**:
    - 按小节或固定时间窗口切片。
    - 统计窗口内的 Pitch Class。
    - 匹配基础三和弦 (Major, Minor, Diminished)。

### 3.3 Analyst Prompt 优化 (`src/agents/analyst.py`)
重构 `_format_prompt_from_data`，将上述数据转化为自然语言描述。

**新增 Prompt 模块**:
- **分轨深度听感**: "贝斯部分十分活跃（密度 4.5 notes/s），覆盖低频区域 (E1-A2)，为乐曲提供了坚实的律动基础..."
- **和声分析**: "乐曲采用了 F# Minor 调性，和弦进行呈现出 [i - VI - III - VII] 的典型流行走向..."
- **人设强化**: 设定为“资深音乐制作人”，要求分析不仅陈述事实，更要关联情感和制作技巧。

## 4. 实施步骤
1.  **修改 Schema**: 更新 `src/schemas.py`。
2.  **实现算法**: 在 `src/agents/transcriber.py` 中填充 `analyze_features` 逻辑。
3.  **更新 Analyst**: 修改 `src/agents/analyst.py` 中的 Prompt 构建逻辑。
4.  **验证**: 使用现有音频跑通流程，检查生成的报告质量。

## 5. 预期效果
最终报告将包含：
- 准确的 BPM 和调性。
- 对每个乐器分轨的具体演奏风格描述（如“稀疏但有力的鼓点”、“跨越两个八度的激昂人声”）。
- 更合理的风格归类理由。
- 类似真人乐评的叙述口吻。
