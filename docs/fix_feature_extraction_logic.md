# 修复与优化特征提取算法方案

## 1. 问题背景
在最近的测试中，系统对不同风格的曲目（如《Shape of You》和《Bohemian Rhapsody》）给出了完全相同的 BPM (136.4)，且调性检测存在不确定性。这表明现有的基于 Basic Pitch 音符事件（Note Events）的简易统计算法存在局限性，可能在音符数据稀疏或量化误差较大时收敛到特定默认值。

## 2. 解决方案目标
利用项目中已安装的 `librosa` 库，引入业界标准的音频信号处理算法，替代或增强原有的 MIDI 统计方法，以获取更准确的音乐特征。

## 3. 技术实施细节

### 3.1 BPM 检测升级 (Audio-based Beat Tracking)
*   **现状**：基于 MIDI 音符起始点间隔（IOI）的直方图统计。
*   **缺陷**：依赖 Basic Pitch 转录的准确性；对稀疏鼓点或复杂切分节奏敏感；Bin 分辨率低。
*   **改进**：
    *   在 `AudioTranscriber` 中引入 `librosa`。
    *   **策略**：优先对 **Drums** 分轨（如果存在且有内容）进行音频加载。
    *   **算法**：使用 `librosa.beat.beat_track` 直接从音频波形中提取 BPM。
    *   **回退**：如果音频分析失败，再回退到原有的 IOI 统计法。

### 3.2 调性检测优化 (Chroma-based Key Detection)
*   **现状**：基于 MIDI 音符时长的 Krumhansl-Schmuckler (K-S) 算法。
*   **缺陷**：Basic Pitch 可能漏掉关键和声内音，导致 Pitch Class 分布偏差。
*   **改进**：
    *   对 **Other** (伴奏) 或 **Piano/Guitar** 分轨进行分析。
    *   **算法**：计算 Chroma 特征 (`librosa.feature.chroma_cqt`)，并在音频层面应用 K-S 轮廓相关性匹配。
    *   **融合**：综合音频检测结果和 MIDI 检测结果（加权投票）。

### 3.3 实施步骤
1.  修改 `src/agents/transcriber.py`:
    *   导入 `librosa`。
    *   新增 `_detect_bpm_from_audio(audio_path)` 方法。
    *   新增 `_detect_key_from_audio(audio_path)` 方法。
    *   在 `analyze_features` 流程中，利用 `midi_data` 中的分轨路径信息，按需加载音频进行高精度分析。
2.  更新 `src/requirements.txt` (确认 `librosa` 已存在)。
3.  验证：重新运行三首曲目的分析，检查 BPM 是否出现差异化（预期 Shape of You ~96, Bohemian Rhapsody ~72-144）。

## 4. 风险控制
*   **性能**：`librosa` 加载音频和计算 CQT 较慢。将限制加载时长（如仅分析中间 60秒）以平衡速度。
*   **依赖**：确保环境支持 `librosa` (已确认在 requirements 中)。

---
**待批准**：请回复“开始”以执行此优化。
