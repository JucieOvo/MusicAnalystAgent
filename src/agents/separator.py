"""
模块名称：separator
功能描述：
    负责音频信号的源分离 (Source Separation)，将混合音频分解为独立的音轨 (Stems)。
    支持人声、鼓、贝斯、吉他、钢琴及其他伴奏的分离。

主要组件：
    - AudioSeparator: 封装 audio-separator 库的核心类，处理模型加载和推理。
    - separate_audio: LangGraph 节点函数，集成到工作流中。

依赖说明：
    - audio-separator: 基于 Demucs/MDX-Net 的分离库
    - torch: 深度学习框架 (由 audio-separator 内部使用)

作者：TraeAI
创建日期：2024-01-XX
修改记录：
    - 2024-01-XX TraeAI: 重构为使用 audio-separator 库，支持 htdemucs_6s 多轨分离。
"""

import shutil
import time
import logging
from pathlib import Path
from typing import Dict, Optional, List

# 抑制 audio_separator 的部分日志，避免刷屏
logging.getLogger("audio_separator").setLevel(logging.WARNING)

from audio_separator.separator import Separator as LibSeparator
from rich.console import Console

from src.config import config, OUTPUT_DIR, MODELS_DIR
from src.schemas import StemType, AnalysisState

console = Console()


class AudioSeparator:
    """
    基于 audio-separator 库的音源分离器
    
    支持模型：
    - htdemucs_6s (默认): 分离 Vocals, Drums, Bass, Guitar, Piano, Other
    - bs_roformer: 高质量人声/伴奏分离
    """
    
    def __init__(self):
        self.config = config.separator
        self.model_name = self.config.model_name
        self.output_format = self.config.output_format
        self._separator: Optional[LibSeparator] = None
        self._loaded = False

    def _init_separator(self):
        """延迟初始化分离引擎"""
        if self._separator is None:
            console.print(f"[cyan]初始化分离引擎 (Model: {self.model_name})...[/cyan]")
            
            # 实例化 Separator
            # 将 config.log_level (str) 转换为 logging 常量 (int)
            log_level_str = config.log_level.upper()
            log_level_int = getattr(logging, log_level_str, logging.INFO)

            # model_file_dir: 指定模型权重下载/加载目录
            self._separator = LibSeparator(
                log_level=log_level_int,
                model_file_dir=str(MODELS_DIR),
                output_dir=None, # 我们将手动处理输出文件移动
                output_format=self.output_format
            )
            
            # 加载模型
            self._separator.load_model(model_filename=self.model_name)
            self._loaded = True
            console.print("[green]分离模型加载完成[/green]")

    def separate(
        self, 
        audio_path: Path,
        output_dir: Optional[Path] = None
    ) -> Dict[StemType, Path]:
        """
        执行音源分离
        
        Args:
            audio_path: 输入音频文件路径
            output_dir: 输出目录，默认为 output/{filename}/stems
            
        Returns:
            分轨路径字典 {StemType: Path}
        """
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
            
        # 设置输出目录
        if output_dir is None:
            if self.config.output_dir:
                 output_dir = self.config.output_dir / audio_path.stem / "stems"
            else:
                 output_dir = OUTPUT_DIR / audio_path.stem / "stems"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化引擎
        self._init_separator()
        
        console.print(f"\n[bold cyan][MUSIC] 开始音源分离[/bold cyan]")
        console.print(f"  输入: {audio_path.name}")
        console.print(f"  输出: {output_dir}")
        console.print(f"  模型: {self.model_name}")
        
        start_time = time.time()
        
        # 执行分离
        # 由于 audio-separator 默认输出到 CWD 或指定 output_dir，且文件名带后缀
        # 我们先让它输出到 output_dir，然后重命名
        
        # 更新 output_dir 配置
        # 注意：LibSeparator 的 output_dir 是在 init 时设置的，或者是 output_dir 属性
        # 但 audio-separator 库结构可能更新，最稳妥是 separate 时传入或者临时修改属性
        # 查看源码或文档，separate 方法通常返回文件列表
        # 我们这里暂时设置 self._separator.output_dir
        self._separator.output_dir = str(output_dir)
        
        console.print("  [INFO] 推理中...")
        output_files = self._separator.separate(str(audio_path))
        
        # 3. 处理输出文件重命名
        console.print("  [INFO] 整理分轨文件...")
        stems_dict = {}
        
        # Stem 名称映射 (文件名关键字 -> StemType)
        # htdemucs_6s 输出通常包含: vocals, drums, bass, guitar, piano, other
        stem_mapping = {
            "vocals": StemType.VOCALS,
            "drums": StemType.DRUMS,
            "bass": StemType.BASS,
            "guitar": StemType.GUITAR,
            "piano": StemType.PIANO,
            "other": StemType.OTHER
        }
        
        for file_path_str in output_files:
            src_path = Path(file_path_str)
            filename_lower = src_path.name.lower()
            
            # 尝试匹配 StemType
            matched_stem = None
            for key, stem_enum in stem_mapping.items():
                # 检查文件名中是否包含 key (例如 "test_(Vocals)_htdemucs_6s.wav" 或 "vocals.wav")
                # 简单的包含检查可能不够，如果有 "other_vocals" 这种奇怪名字
                # 但通常分离器输出比较标准
                if key in filename_lower:
                    matched_stem = stem_enum
                    break
            
            if matched_stem:
                # 目标文件名: vocals.wav, drums.wav ...
                dest_path = output_dir / f"{matched_stem.value}.{self.output_format}"
                
                # 如果是原地重命名 (src 和 dest 在同一目录)
                if src_path != dest_path:
                    # 如果目标已存在，覆盖
                    if dest_path.exists():
                        dest_path.unlink()
                    src_path.rename(dest_path)
                
                stems_dict[matched_stem] = dest_path
            else:
                # 未识别的轨道，保持原名 (已经在 output_dir 中)
                pass
        
        elapsed = time.time() - start_time
        console.print(f"[green][OK] 分离完成[/green] (耗时: {elapsed:.2f}s)")
        
        # 打印结果摘要
        for stem, path in stems_dict.items():
            console.print(f"  - {stem.value}: {path.name}")
            
        return stems_dict


def separate_audio(state: AnalysisState) -> AnalysisState:
    """
    LangGraph 节点函数：音源分离
    
    更新状态中的 stems_paths 字段。
    """
    console.print("\n[bold magenta]=== 听觉分离专家 ===[/bold magenta]")
    
    start_time = time.time()
    
    try:
        separator = AudioSeparator()
        stems = separator.separate(state['audio_path'])
        
        # 更新状态
        # 注意：TypedDict 的更新
        new_state = state.copy()
        new_state['stems_paths'] = stems
        new_state['separation_complete'] = True
        
        # 更新处理时间
        if 'processing_time' not in new_state:
            new_state['processing_time'] = {}
        new_state['processing_time']["separation"] = time.time() - start_time
        
        return new_state
        
    except Exception as e:
        console.print(f"[red][ERROR] 分离失败: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        
        new_state = state.copy()
        if 'errors' not in new_state:
            new_state['errors'] = []
        new_state['errors'].append(f"分离失败: {str(e)}")
        return new_state


# === 命令行测试入口 ===
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python -m src.agents.separator <audio_file>")
        sys.exit(1)
        
    audio_path = Path(sys.argv[1])
    separator = AudioSeparator()
    
    try:
        results = separator.separate(audio_path)
        console.print("\n[bold green]分离结果:[/bold green]")
        for stem_type, path in results.items():
            console.print(f"  {stem_type.value}: {path}")
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
