"""
认知综合层 (The Analyst)
========================

使用 DeepSeek-Reasoner 生成最终的音乐分析报告。
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console
from rich.markdown import Markdown

from src.config import config, OUTPUT_DIR
from src.schemas import AnalysisState, AnalysisResult, MusicalFeatures, SemanticTags

console = Console()


# === 分析报告 Prompt 模板 ===
ANALYST_PROMPT_TEMPLATE = """你是一位资深的音乐制作人和乐评人，拥有敏锐的听觉和深厚的乐理知识。
请根据以下 AI 提取的详细音乐特征数据，撰写一份专业、深度且充满洞见的音乐分析报告。

## 1. 基础信息
- 文件名: {filename}
- 时长: {duration}
- BPM: {bpm}
- 调性: {key}
- 拍号: {time_signature}

## 2. 和声与旋律分析
- 和弦进行: {chord_progression}

## 3. 编曲与配器细节 (分轨深度分析)
{stems_detailed_analysis}

## 4. 语义感知 (AI 听感标签)
- 情感氛围: {mood}
- 风格流派: {genre}
- 音色质感: {texture}
- 识别乐器: {instruments}

---

## 写作要求
请综合上述数据，撰写一份结构清晰的 Markdown 报告。请避免简单罗列数据，而是将数据转化为**音乐性的描述**。

**报告结构：**

1.  **整体听感与风格定位**
    *   结合 BPM、调性和风格标签，描述曲目的整体氛围。
    *   (例如：128 BPM 配合 F# Minor 调性，构建了典型的 Deep House 阴郁而律动的基调...)

2.  **编曲与制作分析**
    *   **核心律动**: 基于 Drums 和 Bass 的密度与活跃度，分析节奏组的表现（如：稀疏的鼓点配合活跃的贝斯线...）。
    *   **旋律与和声**: 分析 Vocals/Other 的音域和密度，以及和弦进行的走向带来的情感张力。
    *   **音响设计**: 结合音色质感标签，评价整体的混音风格（如：Lo-fi 颗粒感、空间感等）。

3.  **情感演进与高潮**
    *   推测音乐的情感发展曲线。

4.  **制作人视角的专业点评**
    *   指出曲目的亮点（如独特的和弦替代、精彩的贝斯编排）。
    *   给出制作上的改进建议。

请保持语气专业、客观但富有感染力，像一位真人在评价这首歌。
"""


class MusicAnalyst:
    """
    音乐分析师
    
    汇总所有专家的分析结果，生成最终报告。
    """
    
    def __init__(self):
        """初始化分析师"""
        self.llm_client = None
        self._initialized = False
        
    def initialize(self) -> None:
        """初始化 LLM 客户端"""
        if self._initialized:
            return
            
        console.print("[cyan]初始化分析师...[/cyan]")
        
        if config.llm.api_key:
            try:
                from openai import OpenAI
                
                self.llm_client = OpenAI(
                    api_key=config.llm.api_key,
                    base_url=config.llm.base_url
                )
                console.print(f"[green]✓ LLM 客户端已连接: {config.llm.model_name}[/green]")
                
            except ImportError:
                console.print("[yellow]⚠ OpenAI 库未安装[/yellow]")
            except Exception as e:
                console.print(f"[red]LLM 初始化失败: {e}[/red]")
        else:
            console.print("[yellow]⚠ 未配置 API Key，将使用模板报告[/yellow]")
            
        self._initialized = True
        
    def generate_report_from_data(
        self,
        audio_path: str,
        stems_paths: Dict[str, str],
        midi_data: Dict[str, Any],
        musical_features: Optional[MusicalFeatures],
        semantic_tags: Optional[SemanticTags]
    ) -> str:
        """
        从分析数据生成报告
        
        Args:
            audio_path: 音频文件路径
            stems_paths: 分轨路径字典
            midi_data: MIDI 数据字典
            musical_features: 音乐特征
            semantic_tags: 语义标签
            
        Returns:
            Markdown 格式的分析报告
        """
        if not self._initialized:
            self.initialize()
            
        # 格式化 Prompt
        prompt = self._format_prompt_from_data(
            audio_path, stems_paths, midi_data, 
            musical_features, semantic_tags
        )
        
        console.print("\n[bold cyan]📝 生成分析报告...[/bold cyan]")
        
        if self.llm_client:
            try:
                response = self.llm_client.chat.completions.create(
                    model=config.llm.model_name,
                    messages=[
                        {"role": "system", "content": "你是一位专业的音乐制作人和乐评人。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_tokens
                )
                
                report = response.choices[0].message.content
                console.print("[green]✓ 报告生成完成[/green]")
                return report
                
            except Exception as e:
                console.print(f"[red]LLM 调用失败: {e}[/red]")
                console.print("[yellow]使用模板报告...[/yellow]")
                
        # 模板报告（当 LLM 不可用时）
        return self._generate_template_report_from_data(
            audio_path, stems_paths, midi_data,
            musical_features, semantic_tags
        )
        
    def _format_prompt_from_data(
        self,
        audio_path: str,
        stems_paths: Dict[str, str],
        midi_data: Dict[str, Any],
        musical_features: Optional[MusicalFeatures],
        semantic_tags: Optional[SemanticTags]
    ) -> str:
        """格式化 Prompt"""
        filename = Path(audio_path).name
        
        # 准备各项数据
        duration = "未知"
        if musical_features and musical_features.duration_seconds:
            mins = int(musical_features.duration_seconds // 60)
            secs = int(musical_features.duration_seconds % 60)
            duration = f"{mins}:{secs:02d}"
            
        bpm = f"{musical_features.bpm:.1f}" if musical_features and musical_features.bpm else "未检测"
        key = musical_features.key if musical_features and musical_features.key else "未检测"
        time_sig = musical_features.time_signature if musical_features else "4/4"
        
        # 和弦进行
        chord_prog = "未检测"
        if musical_features and musical_features.chord_progression:
            # 仅取前 16 个和弦展示，避免 Prompt 过长
            chords = [c.chord_name for c in musical_features.chord_progression[:16]]
            chord_prog = " → ".join(chords)
            if len(musical_features.chord_progression) > 16:
                chord_prog += " ..."
            
        # 分轨详细分析
        stems_detailed_analysis = "暂无详细分轨数据"
        if musical_features and musical_features.stem_analyses:
            lines = []
            for stem_name, analysis in musical_features.stem_analyses.items():
                line = (
                    f"- **{stem_name.upper()}**: {analysis.description}\n"
                    f"  (密度: {analysis.note_density:.1f} notes/s, "
                    f"活跃度: {analysis.active_ratio:.0%}, "
                    f"力度: {analysis.average_velocity:.0f})"
                )
                lines.append(line)
            stems_detailed_analysis = "\n".join(lines)
        elif stems_paths:
             # 回退到简单列表
            stems_list = [f"- {stem}" for stem in stems_paths.keys()]
            stems_detailed_analysis = "\n".join(stems_list)
            
        # 语义标签
        mood = ", ".join(semantic_tags.mood) if semantic_tags else "未分析"
        genre = ", ".join(semantic_tags.genre) if semantic_tags else "未分析"
        texture = ", ".join(semantic_tags.texture) if semantic_tags else "未分析"
        instruments = ", ".join(semantic_tags.instruments) if semantic_tags else "未分析"
        
        return ANALYST_PROMPT_TEMPLATE.format(
            filename=filename,
            duration=duration,
            bpm=bpm,
            key=key,
            time_signature=time_sig,
            chord_progression=chord_prog,
            stems_detailed_analysis=stems_detailed_analysis,
            mood=mood,
            genre=genre,
            texture=texture,
            instruments=instruments
        )
        
    def _generate_template_report_from_data(
        self,
        audio_path: str,
        stems_paths: Dict[str, str],
        midi_data: Dict[str, Any],
        musical_features: Optional[MusicalFeatures],
        semantic_tags: Optional[SemanticTags]
    ) -> str:
        """生成模板报告"""
        filename = Path(audio_path).name
        
        genre_str = ", ".join(semantic_tags.genre) if semantic_tags else "未知"
        mood_str = ", ".join(semantic_tags.mood) if semantic_tags else "未知"
        texture_str = ", ".join(semantic_tags.texture) if semantic_tags else "未知"
        
        duration_str = "未知"
        if musical_features and musical_features.duration_seconds:
            duration_str = f"{musical_features.duration_seconds:.1f}秒"
            
        stems_list = "\n".join([f'- **{stem}**' for stem in stems_paths.keys()]) if stems_paths else "- 未分离"
        
        return f"""# 🎵 音乐分析报告

## 基本信息
- **文件**: {filename}
- **时长**: {duration_str}

## 整体印象
这是一首具有 **{genre_str}** 风格特征的曲目，整体氛围呈现出 **{mood_str}** 的情感色彩。

## 编曲分析
通过音源分离技术，我们识别出以下乐器层：
{stems_list}

## 技术特点
- **节奏结构**: 标准 {musical_features.time_signature if musical_features else '4/4'} 拍
- **调性**: {musical_features.key if musical_features else '待分析'}
- **速度**: {musical_features.bpm if musical_features else '待检测'} BPM

## 语义特征
| 维度 | 标签 |
|------|------|
| 情感 | {mood_str} |
| 风格 | {genre_str} |
| 质感 | {texture_str} |

---
*报告由 Poly-Muse Analyst 自动生成*
"""


def export_result(
    state: AnalysisState, 
    output_path: Optional[Path] = None
) -> AnalysisResult:
    """
    导出最终分析结果
    
    Args:
        state: 完成分析的状态 (TypedDict)
        output_path: 可选的输出文件路径
        
    Returns:
        标准化的分析结果对象
    """
    audio_path = state.get("audio_path", "")
    stems_paths = state.get("stems_paths", {})
    musical_features_dict = state.get("musical_features")
    semantic_tags_dict = state.get("semantic_tags")
    analysis_report = state.get("analysis_report", "")
    
    # 重建 Pydantic 模型
    musical_features = None
    if musical_features_dict:
        musical_features = MusicalFeatures(**musical_features_dict)
        
    semantic_tags = None
    if semantic_tags_dict:
        semantic_tags = SemanticTags(**semantic_tags_dict)
    
    result = AnalysisResult(
        audio_structure={
            "stems_path": stems_paths
        },
        musical_features=musical_features,
        semantic_tags=semantic_tags,
        review=analysis_report
    )
    
    # 保存到文件
    if output_path is None:
        output_path = OUTPUT_DIR / Path(audio_path).stem / "analysis_result.json"
        
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result.model_dump_json(indent=2))
        
    # 同时保存 Markdown 报告
    report_path = output_path.parent / "analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(analysis_report)
        
    console.print(f"\n[green]✓ 结果已保存到: {output_path.parent}[/green]")
    
    return result
