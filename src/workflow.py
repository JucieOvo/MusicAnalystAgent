"""
LangGraph 工作流编排
====================

定义音乐分析的完整工作流状态机。
"""

from typing import Literal
from pathlib import Path

from langgraph.graph import StateGraph, END
from rich.console import Console
from rich.panel import Panel

from src.schemas import AnalysisState

console = Console()


def router_node(state: AnalysisState) -> AnalysisState:
    """
    路由节点：判断任务类型
    """
    console.print(Panel.fit(
        f"[bold cyan]🎵 Poly-Muse Analyst[/bold cyan]\n"
        f"[dim]多模态音乐分析智能体[/dim]",
        border_style="cyan"
    ))
    
    audio_path = state.get("audio_path", "")
    task_type = state.get("task_type", "full_analysis")
    
    console.print(f"\n📂 输入文件: [green]{audio_path}[/green]")
    console.print(f"📋 任务类型: [yellow]{task_type}[/yellow]")
    
    # 验证输入文件
    errors = state.get("errors", [])
    if audio_path and not Path(audio_path).exists():
        errors = errors + [f"文件不存在: {audio_path}"]
        console.print(f"[red]✗ 文件不存在![/red]")
        return {"errors": errors}
        
    return {}  # 不需要更新任何字段


def route_task(state: AnalysisState) -> Literal["full_analysis", "semantic_only", "end"]:
    """
    根据状态决定路由
    """
    errors = state.get("errors", [])
    if errors:
        return "end"
        
    task_type = state.get("task_type", "full_analysis")
    if task_type == "semantic_only":
        return "semantic_only"
        
    return "full_analysis"


def separate_audio(state: AnalysisState) -> AnalysisState:
    """
    LangGraph 节点函数：音源分离
    """
    from src.agents.separator import AudioSeparator
    import time
    
    console.print("\n[bold magenta]═══ 听觉分离专家 ═══[/bold magenta]")
    
    start_time = time.time()
    audio_path = state.get("audio_path", "")
    
    try:
        separator = AudioSeparator()
        stems = separator.separate(Path(audio_path))
        
        # 转换为字符串路径
        stems_paths = {stem.value: str(path) for stem, path in stems.items()}
        
        return {
            "stems_paths": stems_paths,
            "separation_complete": True,
            "processing_time": {"separation": time.time() - start_time}
        }
        
    except Exception as e:
        console.print(f"[red]✗ 分离失败: {e}[/red]")
        return {
            "errors": [f"分离失败: {str(e)}"],
            "separation_complete": False
        }


def transcribe_stems(state: AnalysisState) -> AnalysisState:
    """
    LangGraph 节点函数：符号转录
    """
    from src.agents.transcriber import AudioTranscriber
    from src.schemas import StemType
    from src.config import OUTPUT_DIR
    import time
    
    console.print("\n[bold magenta]═══ 符号转录专家 ═══[/bold magenta]")
    
    separation_complete = state.get("separation_complete", False)
    if not separation_complete:
        console.print("[yellow]⚠ 分离尚未完成，跳过转录[/yellow]")
        return {"transcription_complete": False}
        
    start_time = time.time()
    stems_paths = state.get("stems_paths", {})
    audio_path = state.get("audio_path", "")
    
    try:
        transcriber = AudioTranscriber()
        
        # 转换路径格式
        stems_path_typed = {StemType(k): Path(v) for k, v in stems_paths.items()}
        
        # 设置输出目录
        output_dir = OUTPUT_DIR / Path(audio_path).stem / "midi"
        
        # 转录所有分轨
        midi_results = transcriber.transcribe_all_stems(stems_path_typed, output_dir)
        
        # 分析音乐特征
        features = transcriber.analyze_features(midi_results, stems_path_typed)
        
        # 转换为可序列化格式
        midi_data = {}
        for stem_type, midi in midi_results.items():
            midi_data[stem_type.value] = {
                "notes": [note.model_dump() for note in midi.notes],
                "tempo": midi.tempo
            }
            
        features_dict = features.model_dump() if features else None
        
        return {
            "midi_data": midi_data,
            "musical_features": features_dict,
            "transcription_complete": True,
            "processing_time": {"transcription": time.time() - start_time}
        }
        
    except Exception as e:
        console.print(f"[red]✗ 转录失败: {e}[/red]")
        return {
            "errors": [f"转录失败: {str(e)}"],
            "transcription_complete": False
        }


def analyze_semantics(state: AnalysisState) -> AnalysisState:
    """
    LangGraph 节点函数：语义分析
    """
    from src.agents.semantic_reviewer import SemanticAnalyzer
    import time
    
    console.print("\n[bold magenta]═══ 语义理解专家 ═══[/bold magenta]")
    
    start_time = time.time()
    audio_path = state.get("audio_path", "")
    stems_paths = state.get("stems_paths", {})
    
    try:
        analyzer = SemanticAnalyzer()
        
        # 分析语义
        tags = analyzer.analyze(
            Path(audio_path),
            stems_paths if state.get("separation_complete") else None
        )
        
        return {
            "semantic_tags": tags.model_dump(),
            "semantic_complete": True,
            "processing_time": {"semantic": time.time() - start_time}
        }
        
    except Exception as e:
        console.print(f"[red]✗ 语义分析失败: {e}[/red]")
        return {
            "errors": [f"语义分析失败: {str(e)}"],
            "semantic_complete": False
        }


def generate_analysis(state: AnalysisState) -> AnalysisState:
    """
    LangGraph 节点函数：生成分析报告
    """
    from src.agents.analyst import MusicAnalyst
    from src.schemas import MusicalFeatures, SemanticTags
    import time
    
    console.print("\n[bold magenta]═══ 认知综合层 ═══[/bold magenta]")
    
    start_time = time.time()
    
    try:
        analyst = MusicAnalyst()
        
        # 重构状态用于报告生成
        musical_features = state.get("musical_features")
        semantic_tags = state.get("semantic_tags")
        
        if musical_features:
            musical_features = MusicalFeatures(**musical_features)
        if semantic_tags:
            semantic_tags = SemanticTags(**semantic_tags)
            
        report = analyst.generate_report_from_data(
            audio_path=state.get("audio_path", ""),
            stems_paths=state.get("stems_paths", {}),
            midi_data=state.get("midi_data", {}),
            musical_features=musical_features,
            semantic_tags=semantic_tags
        )
        
        return {
            "analysis_report": report,
            "processing_time": {"analysis": time.time() - start_time}
        }
        
    except Exception as e:
        console.print(f"[red]✗ 报告生成失败: {e}[/red]")
        return {"errors": [f"报告生成失败: {str(e)}"]}


def create_analysis_graph() -> StateGraph:
    """
    创建音乐分析工作流图
    
    工作流结构 (顺序执行以避免并发问题):
    
        [START]
           │
           ▼
      ┌─────────┐
      │ Router  │ ── 判断任务类型
      └────┬────┘
           │
           ▼
      ┌─────────┐
      │Separator│ ── 音源分离 (BS-RoFormer)
      └────┬────┘
           │
           ▼
      ┌─────────┐
      │Transcri │ ── 符号转录 (Basic Pitch)
      └────┬────┘
           │
           ▼
      ┌─────────┐
      │Semantic │ ── 语义分析 (CLaMP 3)
      └────┬────┘
           │
           ▼
      ┌─────────┐
      │ Analyst │ ── 生成报告 (LLM)
      └────┬────┘
           │
           ▼
         [END]
    """
    
    # 创建状态图
    workflow = StateGraph(AnalysisState)
    
    # === 添加节点 ===
    workflow.add_node("router", router_node)
    workflow.add_node("separator", separate_audio)
    workflow.add_node("transcriber", transcribe_stems)
    workflow.add_node("semantic", analyze_semantics)
    workflow.add_node("analyst", generate_analysis)
    
    # === 设置入口 ===
    workflow.set_entry_point("router")
    
    # === 添加边 (顺序执行) ===
    workflow.add_conditional_edges(
        "router",
        route_task,
        {
            "full_analysis": "separator",
            "semantic_only": "semantic",
            "end": END
        }
    )
    
    # 顺序执行: separator -> transcriber -> semantic -> analyst
    workflow.add_edge("separator", "transcriber")
    workflow.add_edge("transcriber", "semantic")
    workflow.add_edge("semantic", "analyst")
    
    # 分析完成后结束
    workflow.add_edge("analyst", END)
    
    return workflow


class MusicAnalysisPipeline:
    """
    音乐分析管道
    
    封装 LangGraph 工作流的高级接口。
    """
    
    def __init__(self):
        """初始化管道"""
        self.graph = create_analysis_graph()
        self.app = self.graph.compile()
        
    def analyze(
        self,
        audio_path: str | Path,
        task_type: str = "full_analysis"
    ) -> AnalysisState:
        """
        执行完整的音乐分析
        
        Args:
            audio_path: 音频文件路径
            task_type: 任务类型 ("full_analysis" | "semantic_only")
            
        Returns:
            完成的分析状态
        """
        # 创建初始状态
        initial_state: AnalysisState = {
            "audio_path": str(Path(audio_path).absolute()),
            "task_type": task_type,
            "stems_paths": {},
            "separation_complete": False,
            "midi_data": {},
            "transcription_complete": False,
            "musical_features": None,
            "semantic_tags": None,
            "semantic_complete": False,
            "analysis_report": None,
            "errors": [],
            "processing_time": {}
        }
        
        # 执行工作流
        console.print("\n[bold]═══════════════════════════════════════[/bold]")
        console.print("[bold cyan]      开始音乐分析工作流[/bold cyan]")
        console.print("[bold]═══════════════════════════════════════[/bold]\n")
        
        import time
        start = time.time()
        
        # 运行图
        final_state = None
        for output in self.app.stream(initial_state):
            # 每个节点的输出
            for node_name, node_state in output.items():
                console.print(f"[dim]完成节点: {node_name}[/dim]")
                final_state = node_state
                
        total_time = time.time() - start
        
        # 打印总结
        console.print("\n[bold]═══════════════════════════════════════[/bold]")
        console.print(f"[bold green]      分析完成! 总耗时: {total_time:.2f}s[/bold green]")
        console.print("[bold]═══════════════════════════════════════[/bold]")
        
        if final_state and final_state.get("processing_time"):
            console.print("\n[dim]各阶段耗时:[/dim]")
            for stage, duration in final_state.get("processing_time", {}).items():
                console.print(f"  {stage}: {duration:.2f}s")
                
        return final_state
        
    def analyze_and_export(
        self,
        audio_path: str | Path,
        output_dir: str | Path | None = None
    ):
        """
        分析并导出结果
        """
        state = self.analyze(audio_path)
        
        if state and state.get("analysis_report"):
            from src.agents.analyst import export_result
            result = export_result(state, output_dir)
            return result
        else:
            console.print("[red]分析未完成，无法导出[/red]")
            return None


# === 简化接口 ===
def analyze_music(audio_path: str | Path) -> AnalysisState:
    """
    快速分析接口
    
    用法:
        from src.workflow import analyze_music
        result = analyze_music("path/to/audio.mp3")
    """
    pipeline = MusicAnalysisPipeline()
    return pipeline.analyze(audio_path)
