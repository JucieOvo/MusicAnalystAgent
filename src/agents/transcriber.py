"""
符号转录专家 (The Transcriber)
==============================

使用 Basic Pitch 将分离后的音轨转换为 MIDI 数据。
"""

import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
from collections import Counter

import numpy as np
import librosa
from rich.console import Console
from rich.table import Table

from src.config import config, OUTPUT_DIR
from src.schemas import (
    StemType, 
    AnalysisState, 
    MIDIData, 
    NoteEvent,
    MusicalFeatures,
    StemAnalysis,
    ChordInfo
)

console = Console()


def calculate_md5(file_path: Path) -> str:
    """计算文件 MD5"""
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        return f"Error: {e}"


class AudioTranscriber:
    """
    Basic Pitch 符号转录器
    
    将分轨音频转录为 MIDI 数据，提取：
    - 音符事件 (Note On/Off, Velocity)
    - BPM 检测
    - 调性分析
    """
    
    def __init__(self):
        """初始化转录器"""
        self.model = None
        self._loaded = False
        
    def load_model(self) -> None:
        """加载 Basic Pitch 模型"""
        if self._loaded:
            return
            
        console.print("[cyan]正在加载 Basic Pitch 模型...[/cyan]")
        
        try:
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH
            
            self._predict = predict
            self._model_path = ICASSP_2022_MODEL_PATH
            self._loaded = True
            console.print("[green]✓ Basic Pitch 模型加载完成[/green]")
            
        except ImportError:
            console.print("[yellow]⚠ Basic Pitch 未安装，使用模拟模式[/yellow]")
            self._loaded = True
            self._predict = None
            
    def transcribe_stem(
        self,
        stem_path: Path,
        stem_type: StemType,
        output_dir: Optional[Path] = None
    ) -> MIDIData:
        """
        转录单个分轨
        
        Args:
            stem_path: 分轨音频路径
            stem_type: 分轨类型
            output_dir: MIDI 输出目录
            
        Returns:
            MIDIData 对象
        """
        if not self._loaded:
            self.load_model()
            
        md5 = calculate_md5(stem_path)
        console.print(f"  转录 {stem_type.value} [dim]({md5[:8]})[/dim]...", end=" ")
        
        notes: List[NoteEvent] = []
        detected_tempo = None
        
        if self._predict is not None and stem_path.exists():
            try:
                # 使用 Basic Pitch 进行转录
                model_output, midi_data, note_events = self._predict(
                    str(stem_path),
                    onset_threshold=config.basic_pitch.onset_threshold,
                    frame_threshold=config.basic_pitch.frame_threshold,
                    minimum_note_length=config.basic_pitch.minimum_note_length,
                    minimum_frequency=config.basic_pitch.minimum_frequency,
                    maximum_frequency=config.basic_pitch.maximum_frequency,
                )
                
                # 转换为我们的数据结构
                for note in note_events:
                    notes.append(NoteEvent(
                        pitch=int(note[2]),  # MIDI pitch
                        start_time=float(note[0]),
                        end_time=float(note[1]),
                        velocity=int(note[3] * 127) if len(note) > 3 else 100
                    ))
                    
                # 保存 MIDI 文件
                if output_dir and midi_data:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    midi_path = output_dir / f"{stem_type.value}.mid"
                    midi_data.write(str(midi_path))
                    
                console.print(f"[green]✓[/green] ({len(notes)} 音符)")
                
            except Exception as e:
                console.print(f"[red]✗ 错误: {e}[/red]")
        else:
            # 模拟模式：生成示例数据
            console.print("[yellow]⚠ 模拟模式[/yellow]")
            
        return MIDIData(
            stem_type=stem_type,
            notes=notes,
            tempo=detected_tempo
        )
        
    def transcribe_all_stems(
        self,
        stems_paths: Dict[StemType, Path],
        output_dir: Optional[Path] = None
    ) -> Dict[StemType, MIDIData]:
        """
        转录所有分轨
        
        针对性转录策略：
        - Drums: 仅提取节奏信息
        - Bass: 仅提取低音线
        - Vocals: 提取旋律线
        - Other: 提取和声信息
        """
        console.print("\n[bold cyan]🎼 开始符号转录[/bold cyan]")
        
        results: Dict[StemType, MIDIData] = {}
        
        for stem_type, stem_path in stems_paths.items():
            midi_data = self.transcribe_stem(stem_path, stem_type, output_dir)
            results[stem_type] = midi_data
            
        return results
        
    def analyze_features(
        self, 
        midi_data: Dict[StemType, MIDIData],
        stems_paths: Optional[Dict[StemType, Path]] = None
    ) -> MusicalFeatures:
        """
        从 MIDI 数据中分析音乐特征
        
        Returns:
            MusicalFeatures 包含 BPM、调性等信息
        """
        console.print("\n[cyan]分析音乐特征...[/cyan]")
        
        # 1. 收集所有音符用于全局分析
        all_notes: List[NoteEvent] = []
        stem_analyses: Dict[str, StemAnalysis] = {}
        
        for stem_type, data in midi_data.items():
            all_notes.extend(data.notes)
            
            # 分轨详细统计
            analysis = self._analyze_stem(stem_type, data.notes)
            stem_analyses[stem_type.value] = analysis
            console.print(f"  - {stem_type.value}: {analysis.description}")

        # 估算总时长
        duration_seconds = 0.0
        if all_notes:
            duration_seconds = max(n.end_time for n in all_notes)

        # 2. 检测 BPM (优先尝试音频检测)
        bpm = None
        if stems_paths:
            # 优先使用 Drums, 其次 Bass, 再次 Mix (如果不在这里的话)
            target_stem = stems_paths.get(StemType.DRUMS) or stems_paths.get(StemType.BASS)
            if target_stem and target_stem.exists():
                bpm = self._detect_bpm_from_audio(target_stem)
                
        if bpm is None:
            bpm = self._detect_bpm(all_notes)
        
        console.print(f"  - 最终 BPM: {bpm:.1f}")

        # 3. 检测调性 (优先尝试音频检测)
        key = None
        if stems_paths:
             # 优先使用和声丰富的轨道
             target_stem = stems_paths.get(StemType.PIANO) or stems_paths.get(StemType.GUITAR) or stems_paths.get(StemType.OTHER)
             if target_stem and target_stem.exists():
                 key = self._detect_key_from_audio(target_stem)
                 
        if key is None:
            key = self._detect_key(all_notes)
            
        console.print(f"  - 最终调性: {key}")

        # 4. 推断和弦进行
        chord_progression = self._infer_chords(all_notes, bpm, duration_seconds)
        if chord_progression:
            prog_str = " -> ".join([c.chord_name for c in chord_progression[:4]])
            console.print(f"  - 和弦进行: {prog_str} ...")

        return MusicalFeatures(
            bpm=bpm,
            key=key,
            time_signature="4/4", # 暂时假定 4/4
            chord_progression=chord_progression,
            duration_seconds=duration_seconds,
            stem_analyses=stem_analyses
        )

    def _analyze_stem(self, stem_type: StemType, notes: List[NoteEvent]) -> StemAnalysis:
        """分析单个分轨的统计特征"""
        if not notes:
            return StemAnalysis(
                stem_type=stem_type.value,
                note_count=0,
                note_density=0.0,
                pitch_range=(0, 0),
                average_velocity=0.0,
                active_ratio=0.0,
                description="无音符数据"
            )

        # 基础统计
        count = len(notes)
        duration = max(n.end_time for n in notes) if notes else 1.0
        pitches = [n.pitch for n in notes]
        velocities = [n.velocity for n in notes]
        
        min_pitch, max_pitch = min(pitches), max(pitches)
        avg_velocity = sum(velocities) / count
        density = count / duration if duration > 0 else 0
        
        # 活跃度计算 (简单的总时值/总时长)
        total_note_duration = sum(n.end_time - n.start_time for n in notes)
        active_ratio = min(1.0, total_note_duration / duration) if duration > 0 else 0

        # 生成描述
        desc_parts = []
        
        # 密度描述
        if density > 8: desc_parts.append("极其密集")
        elif density > 4: desc_parts.append("演奏活跃")
        elif density < 0.5: desc_parts.append("零星点缀")
        else: desc_parts.append("节奏适中")
        
        # 音域描述
        range_semitones = max_pitch - min_pitch
        if range_semitones > 24: desc_parts.append("音域跨度极大")
        elif range_semitones < 7: desc_parts.append("音域集中")
        
        # 力度描述
        if avg_velocity > 100: desc_parts.append("力度强劲")
        elif avg_velocity < 60: desc_parts.append("触键轻柔")

        description = f"{'，'.join(desc_parts)} (范围: {self._midi_to_note_name(min_pitch)}-{self._midi_to_note_name(max_pitch)})"

        return StemAnalysis(
            stem_type=stem_type.value,
            note_count=count,
            note_density=round(density, 2),
            pitch_range=(min_pitch, max_pitch),
            average_velocity=round(avg_velocity, 1),
            active_ratio=round(active_ratio, 2),
            description=description
        )

    def _detect_bpm_from_audio(self, audio_path: Path) -> Optional[float]:
        """从音频中直接检测 BPM"""
        try:
            console.print(f"  [dim]音频 BPM 分析: {audio_path.name}...[/dim]", end=" ")
            # 仅读取前 60 秒以提高速度
            y, sr = librosa.load(str(audio_path), sr=22050, duration=60)
            if len(y) == 0:
                return None
            
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            
            if np.ndim(tempo) > 0:
                tempo = tempo[0]
                
            console.print(f"[green]{tempo:.1f}[/green]")
            return float(tempo)
        except Exception as e:
            console.print(f"[red]失败: {e}[/red]")
            return None

    def _detect_key_from_audio(self, audio_path: Path) -> Optional[str]:
        """从音频中直接检测调性"""
        try:
            console.print(f"  [dim]音频调性分析: {audio_path.name}...[/dim]", end=" ")
            y, sr = librosa.load(str(audio_path), sr=22050, duration=60)
            if len(y) == 0:
                return None
                
            # 计算 Chroma CQT
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_sum = np.sum(chroma, axis=1)
            
            # K-S Profiles
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            
            chroma_norm = chroma_sum / (np.sum(chroma_sum) + 1e-6) * 100
            
            pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            best_corr = -1.0
            best_key = "Unknown"
            
            for i in range(12):
                # Major
                c_maj = np.corrcoef(np.roll(chroma_norm, -i), major_profile)[0, 1]
                if c_maj > best_corr:
                    best_corr = c_maj
                    best_key = f"{pitch_names[i]} Major"
                    
                # Minor
                c_min = np.corrcoef(np.roll(chroma_norm, -i), minor_profile)[0, 1]
                if c_min > best_corr:
                    best_corr = c_min
                    best_key = f"{pitch_names[i]} Minor"
            
            console.print(f"[green]{best_key}[/green]")
            return best_key
        except Exception as e:
            console.print(f"[red]失败: {e}[/red]")
            return None

    def _detect_bpm(self, notes: List[NoteEvent]) -> float:
        """基于 IOI 直方图估算 BPM"""
        if len(notes) < 10:
            return 120.0 # 默认值
            
        # 提取所有 onset
        onsets = sorted([n.start_time for n in notes])
        
        # 计算相邻 IOI (Inter-Onset Intervals)
        iois = []
        for i in range(len(onsets) - 1):
            diff = onsets[i+1] - onsets[i]
            if 0.1 < diff < 2.0: # 过滤掉极短(trill)和极长(pause)的间隔
                iois.append(diff)
                
        if not iois:
            return 120.0
            
        # 直方图统计
        bins = np.arange(0.1, 2.0, 0.02) # 20ms bin
        hist, bin_edges = np.histogram(iois, bins=bins)
        
        # 找到峰值
        peak_idx = np.argmax(hist)
        peak_ioi = (bin_edges[peak_idx] + bin_edges[peak_idx+1]) / 2
        
        bpm = 60.0 / peak_ioi
        
        # 将 BPM 归一化到 70-160 之间
        while bpm < 70: bpm *= 2
        while bpm > 160: bpm /= 2
            
        return round(bpm, 1)

    def _detect_key(self, notes: List[NoteEvent]) -> str:
        """使用 Krumhansl-Schmuckler 算法估算调性"""
        if not notes:
            return "Unknown"
            
        # K-S Profiles
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        
        # 计算 Pitch Class 分布 (按时长加权)
        pc_distribution = [0.0] * 12
        for n in notes:
            pc = n.pitch % 12
            duration = n.end_time - n.start_time
            pc_distribution[pc] += duration
            
        # 归一化
        total_duration = sum(pc_distribution)
        if total_duration == 0: return "Unknown"
        pc_distribution = [x / total_duration * 100 for x in pc_distribution] # 缩放到百分比以便与 profile 匹配
        
        # 计算相关性
        best_corr = -1.0
        best_key = "Unknown"
        
        pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # 遍历 12 个半音移位
        for i in range(12):
            # 测试 Major
            corr_major = np.corrcoef(
                np.roll(pc_distribution, -i), 
                major_profile
            )[0, 1]
            
            if corr_major > best_corr:
                best_corr = corr_major
                best_key = f"{pitch_names[i]} Major"
                
            # 测试 Minor
            corr_minor = np.corrcoef(
                np.roll(pc_distribution, -i), 
                minor_profile
            )[0, 1]
            
            if corr_minor > best_corr:
                best_corr = corr_minor
                best_key = f"{pitch_names[i]} Minor"
                
        return best_key

    def _infer_chords(self, notes: List[NoteEvent], bpm: float, duration: float) -> List[ChordInfo]:
        """简单的和弦推断"""
        if not notes or bpm <= 0:
            return []
            
        seconds_per_beat = 60.0 / bpm
        seconds_per_measure = seconds_per_beat * 4 # 假设 4/4
        
        chords = []
        num_measures = int(math.ceil(duration / seconds_per_measure))
        
        for m in range(num_measures):
            start = m * seconds_per_measure
            end = start + seconds_per_measure
            
            # 获取该小节内的音符
            measure_notes = [n for n in notes if start <= n.start_time < end]
            if not measure_notes:
                continue
                
            # 统计 Pitch Class
            pcs = [n.pitch % 12 for n in measure_notes]
            if not pcs: continue
            
            pc_counts = Counter(pcs)
            root_pc = pc_counts.most_common(1)[0][0]
            
            # 简单判断大/小三和弦
            has_major_3rd = (root_pc + 4) % 12 in pcs
            has_minor_3rd = (root_pc + 3) % 12 in pcs
            
            chord_name = self._midi_to_note_name(root_pc, with_octave=False)
            if has_minor_3rd:
                chord_name += "m"
            elif not has_major_3rd:
                # 既无大三也无小三，可能是挂留或仅仅是单音，暂且只标根音
                pass
                
            chords.append(ChordInfo(
                chord_name=chord_name,
                start_time=round(start, 2),
                end_time=round(end, 2)
            ))
            
        return chords

    def _midi_to_note_name(self, midi: int, with_octave: bool = True) -> str:
        """MIDI 音高转音名"""
        names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        pitch = midi % 12
        octave = midi // 12 - 1
        name = names[pitch]
        return f"{name}{octave}" if with_octave else name


def transcribe_stems(state: AnalysisState) -> AnalysisState:
    """
    LangGraph 节点函数：符号转录
    
    依赖于分离节点完成后运行。
    更新状态中的 midi_data 和 musical_features 字段。
    """
    console.print("\n[bold magenta]═══ 符号转录专家 ═══[/bold magenta]")
    
    if not state.separation_complete:
        console.print("[yellow]⚠ 分离尚未完成，跳过转录[/yellow]")
        return state
        
    start_time = time.time()
    
    try:
        transcriber = AudioTranscriber()
        
        # 设置输出目录
        output_dir = OUTPUT_DIR / state.audio_path.stem / "midi"
        
        # 转录所有分轨
        midi_results = transcriber.transcribe_all_stems(
            state.stems_paths,
            output_dir
        )
        
        # 分析音乐特征
        features = transcriber.analyze_features(midi_results, state.stems_paths)
        
        # 更新状态
        state.midi_data = midi_results
        state.musical_features = features
        state.transcription_complete = True
        state.processing_time["transcription"] = time.time() - start_time
        
        # 打印摘要
        table = Table(title="转录结果摘要")
        table.add_column("分轨", style="cyan")
        table.add_column("音符数", justify="right")
        
        for stem_type, midi in midi_results.items():
            table.add_row(stem_type.value, str(len(midi.notes)))
            
        console.print(table)
        
    except Exception as e:
        state.errors.append(f"转录失败: {str(e)}")
        console.print(f"[red]✗ 转录失败: {e}[/red]")
        
    return state


# === 命令行测试入口 ===
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python transcriber.py <stem_file>")
        sys.exit(1)
        
    stem_file = Path(sys.argv[1])
    transcriber = AudioTranscriber()
    
    try:
        result = transcriber.transcribe_stem(stem_file, StemType.VOCALS)
        console.print(f"\n[bold green]转录完成:[/bold green]")
        console.print(f"  音符数: {len(result.notes)}")
        if result.notes:
            console.print(f"  首个音符: pitch={result.notes[0].pitch}, "
                         f"start={result.notes[0].start_time:.2f}s")
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")
        sys.exit(1)
