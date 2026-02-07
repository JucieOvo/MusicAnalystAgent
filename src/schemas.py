"""
数据模式定义
============

定义系统中流转的所有数据结构。
"""

from typing import Dict, List, Optional, Annotated, Any
from typing_extensions import TypedDict
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum
import operator


class StemType(str, Enum):
    """分轨类型枚举"""
    VOCALS = "vocals"
    DRUMS = "drums"
    BASS = "bass"
    OTHER = "other"
    PIANO = "piano"
    GUITAR = "guitar"


class NoteEvent(BaseModel):
    """单个音符事件"""
    pitch: int = Field(description="MIDI音高 (0-127)")
    start_time: float = Field(description="起始时间(秒)")
    end_time: float = Field(description="结束时间(秒)")
    velocity: int = Field(default=100, description="力度 (0-127)")


class MIDIData(BaseModel):
    """MIDI数据结构"""
    stem_type: StemType = Field(description="对应的分轨类型")
    notes: List[NoteEvent] = Field(default_factory=list, description="音符事件列表")
    tempo: Optional[float] = Field(default=None, description="检测到的BPM")
    

class ChordInfo(BaseModel):
    """和弦信息"""
    chord_name: str = Field(description="和弦名称，如 Cm, G7")
    start_time: float = Field(description="起始时间")
    end_time: float = Field(description="结束时间")
    roman_numeral: Optional[str] = Field(default=None, description="级数表示，如 vi, IV")


class StemAnalysis(BaseModel):
    """分轨详细分析"""
    stem_type: str = Field(description="分轨类型")
    note_count: int = Field(description="音符总数")
    note_density: float = Field(description="音符密度(个/秒)")
    pitch_range: tuple[int, int] = Field(description="音域范围 (min, max)")
    average_velocity: float = Field(description="平均力度")
    active_ratio: float = Field(description="活跃时间比例 (0.0-1.0)")
    description: str = Field(default="", description="基于统计的简短描述")


class MusicalFeatures(BaseModel):
    """音乐特征结构"""
    bpm: Optional[float] = Field(default=None, description="节拍速度")
    key: Optional[str] = Field(default=None, description="调性，如 'F# Minor'")
    time_signature: Optional[str] = Field(default="4/4", description="拍号")
    chord_progression: List[ChordInfo] = Field(
        default_factory=list, 
        description="和弦进行"
    )
    duration_seconds: Optional[float] = Field(default=None, description="时长(秒)")
    stem_analyses: Dict[str, StemAnalysis] = Field(
        default_factory=dict,
        description="各分轨的详细统计分析"
    )


class SemanticTags(BaseModel):
    """语义标签集合"""
    mood: List[str] = Field(default_factory=list, description="情感标签")
    genre: List[str] = Field(default_factory=list, description="风格流派")
    instruments: List[str] = Field(default_factory=list, description="乐器识别")
    texture: List[str] = Field(default_factory=list, description="音色质感")
    era: Optional[str] = Field(default=None, description="时代风格")
    
    # 置信度分数
    confidence_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="各标签的置信度"
    )


# === Reducer 函数用于合并并行更新 ===
def merge_dicts(left: Dict, right: Dict) -> Dict:
    """合并两个字典"""
    if left is None:
        return right
    if right is None:
        return left
    return {**left, **right}


def keep_last(left: Any, right: Any) -> Any:
    """保留最后一个非空值"""
    return right if right is not None else left


def merge_errors(left: List[str], right: List[str]) -> List[str]:
    """合并错误列表"""
    if left is None:
        left = []
    if right is None:
        right = []
    return left + right


class AnalysisState(TypedDict, total=False):
    """
    全局分析状态 - LangGraph State Schema
    ======================================
    
    使用 TypedDict 和 Annotated 支持并行节点更新。
    """
    # === 输入 (不变) ===
    audio_path: str  # 使用 str 而不是 Path，因为 TypedDict 序列化更简单
    task_type: str
    
    # === 分离层输出 ===
    stems_paths: Annotated[Dict[str, str], merge_dicts]
    separation_complete: bool
    
    # === 转录层输出 ===
    midi_data: Annotated[Dict[str, Any], merge_dicts]
    transcription_complete: bool
    
    # === 特征分析 ===
    musical_features: Annotated[Optional[Dict], keep_last]
    
    # === 语义层输出 ===
    semantic_tags: Annotated[Optional[Dict], keep_last]
    semantic_complete: bool
    
    # === 最终输出 ===
    analysis_report: Annotated[Optional[str], keep_last]
    
    # === 元数据 ===
    errors: Annotated[List[str], merge_errors]
    processing_time: Annotated[Dict[str, float], merge_dicts]


# === Pydantic 模型用于验证和序列化 ===
class AnalysisStateModel(BaseModel):
    """
    AnalysisState 的 Pydantic 版本，用于验证和 JSON 序列化
    """
    audio_path: Path = Field(description="原始音频文件路径")
    task_type: str = Field(default="full_analysis", description="任务类型")
    
    stems_paths: Dict[str, str] = Field(default_factory=dict)
    separation_complete: bool = Field(default=False)
    
    midi_data: Dict[str, Any] = Field(default_factory=dict)
    transcription_complete: bool = Field(default=False)
    
    musical_features: Optional[MusicalFeatures] = Field(default=None)
    semantic_tags: Optional[SemanticTags] = Field(default=None)
    semantic_complete: bool = Field(default=False)
    
    analysis_report: Optional[str] = Field(default=None)
    
    errors: List[str] = Field(default_factory=list)
    processing_time: Dict[str, float] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    """最终分析结果（用于API输出）"""
    audio_structure: Dict[str, Any] = Field(description="音频结构信息")
    musical_features: Optional[MusicalFeatures] = None
    semantic_tags: Optional[SemanticTags] = None
    review: str = Field(description="生成的分析报告")
    
    class Config:
        json_schema_extra = {
            "example": {
                "audio_structure": {
                    "stems_path": {
                        "vocals": "/path/to/vocal.wav",
                        "drums": "/path/to/drums.wav"
                    }
                },
                "musical_features": {
                    "bpm": 124,
                    "key": "F# Minor",
                    "chord_progression": ["vi", "IV", "I", "V"]
                },
                "semantic_tags": {
                    "mood": ["Energetic", "Tense"],
                    "genre": ["Synthwave", "Cyberpunk"]
                },
                "review": "本曲目展现了典型的 Synthwave 风格..."
            }
        }
