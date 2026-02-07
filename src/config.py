"""
全局配置模块
============

管理模型路径、API密钥和系统参数。
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 加载环境变量
load_dotenv()

# === 路径配置 ===
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
DESCRIPTOR_BANK_PATH = PROJECT_ROOT / "data" / "descriptor_bank.json"

# 确保输出目录存在
OUTPUT_DIR.mkdir(exist_ok=True)


class BSRoformerConfig(BaseModel):
    """BS-RoFormer 音源分离模型配置"""
    checkpoint_path: Path = Field(
        default=MODELS_DIR / "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        description="模型权重文件路径"
    )
    config_path: Path = Field(
        default=MODELS_DIR / "model_bs_roformer_ep_317_sdr_12.9755.yaml",
        description="模型配置文件路径"
    )
    device: str = Field(
        default="cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu",
        description="推理设备"
    )
    use_fp16: bool = Field(
        default=True,
        description="是否使用半精度推理加速"
    )


class SeparatorConfig(BaseModel):
    """Audio Separator 配置"""
    model_name: str = Field(default="htdemucs_6s.yaml", description="分离模型名称")
    output_format: str = Field(default="wav", description="输出格式")
    use_gpu: bool = Field(default=True, description="是否使用 GPU")
    output_dir: Optional[Path] = Field(default=None, description="自定义输出目录")


class BasicPitchConfig(BaseModel):
    """Basic Pitch 符号转录配置"""
    onset_threshold: float = Field(default=0.5, description="音符起始检测阈值")
    frame_threshold: float = Field(default=0.3, description="帧级别激活阈值")
    minimum_note_length: float = Field(default=58.0, description="最小音符长度(ms)")
    minimum_frequency: float = Field(default=32.7, description="最低频率(Hz)")
    maximum_frequency: float = Field(default=2000.0, description="最高频率(Hz)")


class LLMConfig(BaseModel):
    """大语言模型配置"""
    api_key: str = Field(
        default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", ""),
        description="DeepSeek API Key"
    )
    base_url: str = Field(
        default="https://api.deepseek.com",
        description="API Base URL"
    )
    model_name: str = Field(
        default="deepseek-reasoner",
        description="模型名称"
    )
    temperature: float = Field(default=0.7, description="生成温度")
    max_tokens: int = Field(default=4096, description="最大输出token数")


class AppConfig(BaseModel):
    """应用全局配置"""
    bs_roformer: BSRoformerConfig = Field(default_factory=BSRoformerConfig)
    separator: SeparatorConfig = Field(default_factory=SeparatorConfig)
    basic_pitch: BasicPitchConfig = Field(default_factory=BasicPitchConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    
    # 采样率设置
    sample_rate: int = Field(default=44100, description="音频采样率")
    
    # 日志级别
    log_level: str = Field(default="INFO", description="日志级别")


# 全局配置实例
config = AppConfig()
