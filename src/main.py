"""
Poly-Muse Analyst 命令行入口
============================

用法:
    python -m src.main analyze <audio_file>
    python -m src.main analyze <audio_file> --output <output_dir>
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from src.config import config, OUTPUT_DIR
from src.workflow import MusicAnalysisPipeline, analyze_music

app = typer.Typer(
    name="poly-muse",
    help="🎵 Poly-Muse Analyst - 多模态音乐分析智能体"
)
console = Console()


@app.command()
def analyze(
    audio_file: Path = typer.Argument(
        ...,
        help="要分析的音频文件路径",
        exists=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="输出目录路径"
    ),
    task_type: str = typer.Option(
        "full_analysis",
        "--task", "-t",
        help="任务类型: full_analysis | semantic_only"
    ),
    export_json: bool = typer.Option(
        True,
        "--export/--no-export",
        help="是否导出 JSON 结果"
    )
):
    """
    分析音频文件并生成报告
    """
    console.print(Panel.fit(
        "[bold cyan]🎵 Poly-Muse Analyst[/bold cyan]\n"
        "[dim]多模态音乐分析智能体 v0.1.0[/dim]",
        border_style="cyan"
    ))
    
    try:
        pipeline = MusicAnalysisPipeline()
        state = pipeline.analyze(audio_file, task_type)
        
        if export_json and state and state.get("analysis_report"):
            from src.agents.analyst import export_result
            output_path = output / "analysis_result.json" if output else None
            export_result(state, output_path)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]分析已取消[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]分析失败: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@app.command()
def separate(
    audio_file: Path = typer.Argument(
        ...,
        help="要分离的音频文件",
        exists=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="输出目录"
    )
):
    """
    仅执行音源分离
    """
    from src.agents.separator import AudioSeparator
    
    separator = AudioSeparator()
    stems = separator.separate(audio_file, output)
    
    console.print("\n[bold green]分离完成![/bold green]")
    for stem_type, path in stems.items():
        console.print(f"  {stem_type.value}: {path}")


@app.command()
def transcribe(
    stem_file: Path = typer.Argument(
        ...,
        help="要转录的音轨文件",
        exists=True
    ),
    stem_type: str = typer.Option(
        "vocals",
        "--type", "-t",
        help="音轨类型: vocals | drums | bass | other"
    )
):
    """
    将音轨转录为 MIDI
    """
    from src.agents.transcriber import AudioTranscriber
    from src.schemas import StemType
    
    transcriber = AudioTranscriber()
    midi_data = transcriber.transcribe_stem(
        stem_file, 
        StemType(stem_type)
    )
    
    console.print(f"\n[bold green]转录完成![/bold green]")
    console.print(f"  音符数: {len(midi_data.notes)}")


@app.command()
def init_descriptors():
    """
    初始化描述符库
    """
    from src.agents.semantic_reviewer import init_descriptor_bank
    init_descriptor_bank()


@app.command()
def version():
    """
    显示版本信息
    """
    from src import __version__
    console.print(f"Poly-Muse Analyst v{__version__}")


@app.command()
def info():
    """
    显示系统配置信息
    """
    from rich.table import Table
    
    table = Table(title="系统配置")
    table.add_column("配置项", style="cyan")
    table.add_column("值", style="green")
    
    table.add_row("BS-RoFormer 权重", str(config.bs_roformer.checkpoint_path))
    table.add_row("推理设备", config.bs_roformer.device)
    table.add_row("FP16 加速", "是" if config.bs_roformer.use_fp16 else "否")
    table.add_row("LLM 模型", config.llm.model_name)
    table.add_row("API Key 已配置", "是" if config.llm.api_key else "否")
    table.add_row("采样率", f"{config.sample_rate} Hz")
    table.add_row("输出目录", str(OUTPUT_DIR))
    
    console.print(table)


def main():
    """主入口"""
    app()


if __name__ == "__main__":
    main()
