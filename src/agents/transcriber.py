"""
符号转录专家 (The Transcriber)
==============================

使用 Basic Pitch 将分离后的音轨转换为 MIDI 数据。
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table

from src.config import config, OUTPUT_DIR
from src.schemas import (
    StemType, 
    AnalysisState, 
    MIDIData, 
    NoteEvent,
    MusicalFeatures
)

console = Console()


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
            
        console.print(f"  转录 {stem_type.value}...", end=" ")
        
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
        midi_data: Dict[StemType, MIDIData]
    ) -> MusicalFeatures:
        """
        从 MIDI 数据中分析音乐特征
        
        Returns:
            MusicalFeatures 包含 BPM、调性等信息
        """
        console.print("\n[cyan]分析音乐特征...[/cyan]")
        
        # TODO: 实现实际的特征分析
        # - BPM 检测：分析鼓轨的节奏密度
        # - 调性检测：分析 Bass 和 Other 轨的音高分布
        # - 和弦进行：根据同时发声的音符推断
        
        features = MusicalFeatures(
            bpm=None,
            key=None,
            time_signature="4/4",
            chord_progression=[],
            duration_seconds=None
        )
        
        # 从所有音符中估算时长
        all_end_times = []
        for midi in midi_data.values():
            for note in midi.notes:
                all_end_times.append(note.end_time)
                
        if all_end_times:
            features.duration_seconds = max(all_end_times)
            
        return features


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
        features = transcriber.analyze_features(midi_results)
        
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
