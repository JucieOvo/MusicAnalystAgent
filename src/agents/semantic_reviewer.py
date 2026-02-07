"""
语义理解专家 (The Semantic Reviewer)
====================================

使用 RAG-Reviewer 架构进行音乐语义分析：
- 描述符库检索
- CLaMP 3 音频编码 (Native Implementation)
- 语义标签提取
"""

import time
import json
import torch
import torchaudio
import numpy as np
import requests
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

from rich.console import Console
from rich.table import Table

# 添加 lib 路径以导入 CLaMP 3 原生模块
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
LIB_DIR = PROJECT_ROOT_DIR / "lib" / "clamp3"
CODE_DIR = LIB_DIR / "code"
AUDIO_PREPROC_DIR = LIB_DIR / "preprocessing" / "audio"

sys.path.append(str(CODE_DIR))
sys.path.append(str(AUDIO_PREPROC_DIR))

# CLaMP 3 Native Imports
try:
    from config import (
        CLAMP3_WEIGHTS_PATH, 
        AUDIO_HIDDEN_SIZE, 
        AUDIO_NUM_LAYERS, 
        MAX_AUDIO_LENGTH,
        M3_HIDDEN_SIZE, 
        PATCH_NUM_LAYERS, 
        PATCH_LENGTH, 
        TEXT_MODEL_NAME, 
        CLAMP3_HIDDEN_SIZE, 
        CLAMP3_LOAD_M3,
        MAX_TEXT_LENGTH
    )
    from utils import CLaMP3Model
    from hf_pretrains import HuBERTFeature
    from transformers import BertConfig, AutoTokenizer
except ImportError as e:
    # 首次运行时可能因路径问题报错，此时无法继续
    print(f"Error importing CLaMP 3 modules: {e}")
    # Define dummy variables to avoid linter errors before sys.path takes effect in runtime
    CLAMP3_WEIGHTS_PATH = ""

from src.config import config as app_config, PROJECT_ROOT, DESCRIPTOR_BANK_PATH
from src.schemas import AnalysisState, SemanticTags

console = Console()

# === 默认描述符库 ===
DEFAULT_DESCRIPTORS = {
    "mood": [
        "Melancholic", "Euphoric", "Nostalgic", "Aggressive", "Peaceful",
        "Tense", "Hopeful", "Dark", "Uplifting", "Dreamy", "Energetic",
        "Sad", "Happy", "Anxious", "Calm", "Powerful", "Romantic"
    ],
    "genre": [
        "Pop", "Rock", "Jazz", "Classical", "Electronic", "Hip-Hop",
        "R&B", "Country", "Folk", "Metal", "Punk", "Blues", "Soul",
        "Funk", "Reggae", "Latin", "World", "Ambient", "Techno",
        "House", "Synthwave", "Vaporwave", "Lo-fi", "Trap"
    ],
    "instruments": [
        "Piano", "Guitar", "Drums", "Bass", "Violin", "Cello",
        "Synthesizer", "Trumpet", "Saxophone", "Flute", "Organ",
        "Strings", "Brass", "Woodwinds", "Percussion", "Vocals"
    ],
    "texture": [
        "Distorted", "Clean", "Reverb-heavy", "Dry", "Lo-fi",
        "Hi-fi", "Warm", "Cold", "Bright", "Dark", "Muddy",
        "Crisp", "Fuzzy", "Ethereal", "Gritty", "Smooth"
    ],
    "era": [
        "60s", "70s", "80s", "90s", "2000s", "2010s", "Modern",
        "Retro", "Vintage", "Contemporary", "Futuristic"
    ]
}


class DescriptorBank:
    """
    描述符库 - 静态资产与 Embedding 缓存
    
    管理音乐术语及其对应的 CLaMP 文本向量。
    """
    
    def __init__(self, bank_path: Optional[Path] = None):
        """
        初始化描述符库
        
        Args:
            bank_path: 描述符库 JSON 文件路径
        """
        self.bank_path = bank_path or DESCRIPTOR_BANK_PATH
        self.cache_path = self.bank_path.parent / "descriptor_embeddings_clamp3.npy"
        
        self.descriptors: Dict[str, List[str]] = {}
        # 平铺的标签列表，用于批量计算
        self.flat_tags: List[Tuple[str, str]] = []  # (category, tag)
        self.embeddings: Optional[np.ndarray] = None # (N, D)
        
        self._loaded = False
        
    def load(self) -> None:
        """加载描述符 JSON 及其 Embeddings (如果存在)"""
        if self._loaded:
            return
            
        # 1. 加载 JSON
        if self.bank_path.exists():
            console.print(f"[cyan]加载描述符库: {self.bank_path}[/cyan]")
            with open(self.bank_path, 'r', encoding='utf-8') as f:
                self.descriptors = json.load(f)
        else:
            console.print("[yellow]使用默认描述符库[/yellow]")
            self.descriptors = DEFAULT_DESCRIPTORS
            # 自动保存默认库
            self.save_default()
            
        # 2. 构建平铺列表
        self.flat_tags = []
        for category, tag_list in self.descriptors.items():
            for tag in tag_list:
                self.flat_tags.append((category, tag))
                
        # 3. 尝试加载缓存的 Embeddings
        if self.cache_path.exists():
            try:
                self.embeddings = np.load(self.cache_path)
                console.print(f"[green]✓ 已加载 Embedding 缓存: {self.embeddings.shape}[/green]")
                
                # 校验缓存大小是否匹配
                if self.embeddings.shape[0] != len(self.flat_tags):
                    console.print("[yellow]⚠ 缓存大小不匹配，将重新计算[/yellow]")
                    self.embeddings = None
            except Exception as e:
                console.print(f"[red]加载缓存失败: {e}[/red]")
                self.embeddings = None
        
        total = len(self.flat_tags)
        console.print(f"[green]✓ 描述符就绪: {total} 个标签[/green]")
        self._loaded = True
    
    def compute_missings(self, analyzer: 'SemanticAnalyzer') -> None:
        """
        计算缺失的 Embeddings
        
        Args:
            analyzer: SemanticAnalyzer instance
        """
        if self.embeddings is not None:
            return
            
        console.print("[cyan]正在计算描述符 Embeddings (CLaMP 3)...[/cyan]")
        
        texts = [f"A music track with {tag} {category}" for category, tag in self.flat_tags]
        
        # 分批处理以避免 OOM
        batch_size = 32
        all_embeds = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # 使用 Analyzer 的 encode_text 方法
            text_features = analyzer.encode_text_batch(batch_texts)
            if text_features is not None:
                all_embeds.append(text_features)
            else:
                raise RuntimeError("Failed to encode text batch")
                
        self.embeddings = np.vstack(all_embeds)
        
        # 保存缓存
        np.save(self.cache_path, self.embeddings)
        console.print(f"[green]✓ Embeddings 已计算并保存: {self.embeddings.shape}[/green]")

    def save_default(self) -> None:
        """保存默认描述符库到文件"""
        self.bank_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.bank_path, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_DESCRIPTORS, f, indent=2, ensure_ascii=False)


class SemanticAnalyzer:
    """
    语义分析器 (Native CLaMP 3)
    
    使用原生 CLaMP 3 实现进行音频-文本跨模态检索。
    流程：Audio -> MERT (m-a-p/MERT-v1-95M) -> CLaMP 3 Encoder -> Global Semantic Vector
    """
    
    def __init__(self):
        """初始化分析器"""
        self.clamp_model = None
        self.mert_model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.descriptor_bank = DescriptorBank()
        self._loaded = False
        
        # MERT 配置
        self.target_sr = 24000
        self.mert_model_name = "m-a-p/MERT-v1-95M"
        
    def _check_and_download_weights(self):
        """检查并下载 CLaMP 3 权重"""
        # 修正路径：确保使用 config.py 中定义的相对路径的绝对位置
        # config.py 中的 CLAMP3_WEIGHTS_PATH 是相对路径 (e.g., "weights_clamp3_saas...")
        # 我们假设它位于 lib/clamp3/code 目录下
        weights_path = CODE_DIR / CLAMP3_WEIGHTS_PATH
        
        if not weights_path.exists():
            console.print(f"[yellow]CLaMP 3 权重未找到，正在下载...[/yellow]")
            console.print(f"目标路径: {weights_path}")
            
            url = "https://huggingface.co/sander-wood/clamp3/resolve/main/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
            
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                with open(weights_path, "wb") as f, tqdm(
                    desc="Downloading",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
                console.print("[green]✓ CLaMP 3 权重下载完成[/green]")
            except Exception as e:
                console.print(f"[red]下载权重失败: {e}[/red]")
                raise e
        
        return weights_path

    def load_model(self) -> None:
        """加载 CLaMP 3 和 MERT 模型"""
        if self._loaded:
            return
            
        console.print(f"[cyan]正在加载语义分析模型 (Device: {self.device})...[/cyan]")
        
        try:
            # 1. 准备权重
            weights_path = self._check_and_download_weights()
            
            # 初始化 MERT 模型 (用于音频特征提取)
            console.print(f"[yellow]Loading MERT model: {self.mert_model_name}...[/yellow]")
            # HuBERTFeature(pre_trained_folder, sample_rate, ...)
            self.mert_model = HuBERTFeature(self.mert_model_name, 24000)
            self.mert_model.to(self.device)
            self.mert_model.eval()
            console.print("[green]✓ MERT model loaded[/green]")
            
            # 3. 加载 CLaMP 3 (Semantic Encoder)
            console.print(f"[cyan]加载 CLaMP 3 模型...[/cyan]")
            
            # 配置
            audio_config = BertConfig(
                vocab_size=1,
                hidden_size=AUDIO_HIDDEN_SIZE,
                num_hidden_layers=AUDIO_NUM_LAYERS,
                num_attention_heads=AUDIO_HIDDEN_SIZE//64,
                intermediate_size=AUDIO_HIDDEN_SIZE*4,
                max_position_embeddings=MAX_AUDIO_LENGTH
            )
            symbolic_config = BertConfig(
                vocab_size=1,
                hidden_size=M3_HIDDEN_SIZE,
                num_hidden_layers=PATCH_NUM_LAYERS,
                num_attention_heads=M3_HIDDEN_SIZE//64,
                intermediate_size=M3_HIDDEN_SIZE*4,
                max_position_embeddings=PATCH_LENGTH
            )
            
            # 初始化模型
            self.clamp_model = CLaMP3Model(
                audio_config=audio_config,
                symbolic_config=symbolic_config,
                text_model_name=TEXT_MODEL_NAME,
                hidden_size=CLAMP3_HIDDEN_SIZE,
                load_m3=False # 推理时不需要加载 M3 训练权重，只需要加载 CLaMP3 整体权重
            )
            
            # 加载权重
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.clamp_model.load_state_dict(checkpoint['model'])
            console.print(f"Loaded CLaMP 3 Checkpoint from Epoch {checkpoint.get('epoch', '?')} with loss {checkpoint.get('min_eval_loss', '?')}")
            
            self.clamp_model.to(self.device)
            self.clamp_model.eval()
            
            # 4. 加载 Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
            
            # 5. 加载描述符库并计算缺失的 Embeddings
            self._loaded = True
            self.descriptor_bank.load()
            self.descriptor_bank.compute_missings(self)
            
            console.print(f"[green]✓ 所有模型加载完成[/green]")
            
        except Exception as e:
            console.print(f"[red]模型加载失败: {e}[/red]")
            import traceback
            console.print(traceback.format_exc())
            raise e

    def encode_text_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        批量编码文本
        """
        if not self._loaded:
            return None
            
        try:
            # Tokenize
            # 使用 tokenizer.sep_token 连接不需要，因为这里是 list of strings
            # 但 extract_clamp3.py 中处理单个 txt 文件是 join 后 tokenize，这里我们处理 list
            # CLaMP 3 的 tokenizer 是 XLM-R
            
            # 这里的处理逻辑参考 extract_clamp3.py 中 .txt 文件的处理，但它是处理一个长文本。
            # 我们是处理 batch of short texts (tags).
            
            # 对每个文本进行 tokenize 和 padding
            encoded_input = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=MAX_TEXT_LENGTH, 
                return_tensors="pt"
            )
            
            input_ids = encoded_input['input_ids'].to(self.device)
            attention_mask = encoded_input['attention_mask'].to(self.device)
            
            with torch.no_grad():
                text_features = self.clamp_model.get_text_features(
                    text_inputs=input_ids,
                    text_masks=attention_mask,
                    get_global=True
                )
                
                # 归一化
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
            return text_features.cpu().numpy()
            
        except Exception as e:
            console.print(f"[red]文本编码失败: {e}[/red]")
            return None

    def encode_audio(self, audio_path: Path) -> Optional[np.ndarray]:
        """
        计算音频的 CLaMP Embedding
        
        流程:
        1. Load & Resample (24k)
        2. Split into 5s chunks
        3. MERT Feature Extraction -> Mean Pooling
        4. Concatenate chunks -> CLaMP 3 Audio Encoder -> Global Vector
        """
        if not self._loaded:
            return None
            
        try:
            # 1. 加载音频
            waveform, sr = torchaudio.load(str(audio_path))
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample to 24k for MERT
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = resampler(waveform)
                sr = self.target_sr
                
            # 2. 切分 (5s windows)
            window_size = 5 # seconds
            window_samples = int(window_size * sr)
            
            # Pad if needed
            if waveform.shape[1] < window_samples:
                pad_len = window_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            
            # Split into chunks (no overlap as per extract_mert.py logic implied by sliding_window_overlap_in_percent=0.0 default)
            chunks = []
            for i in range(0, waveform.shape[1], window_samples):
                chunk = waveform[:, i:i+window_samples]
                if chunk.shape[1] < window_samples:
                    # Pad last chunk
                    pad_len = window_samples - chunk.shape[1]
                    chunk = torch.nn.functional.pad(chunk, (0, pad_len))
                chunks.append(chunk)
            
            if not chunks:
                return None
                
            chunks = torch.stack(chunks).to(self.device) # (N, 1, samples)
            
            # 3. MERT 提取
            mert_features_list = []
            with torch.no_grad():
                for i in range(chunks.shape[0]):
                    # HuBERTFeature expects (B, T) input, our chunks are (1, 1, samples) -> (1, samples)
                    wav_input = chunks[i] # (1, samples)
                    
                    # process_wav does padding/norm
                    wav_input = self.mert_model.process_wav(wav_input).to(self.device)
                    
                    # forward(input_values, layer=None, reduction="mean")
                    # layer=None means all layers, reduction="mean" means average over time
                    # But extract_mert.py uses layer=None, reduction="mean" -> returns [L, B, H]
                    # Wait, let's check extract_mert.py line 98: if mean_features: features = features.mean(dim=0, keepdim=True)
                    # extract_mert.py default reduction is 'mean'.
                    # In extract_mert.py:
                    # features = feature_extractor(wav_chunk, layer=layer, reduction=reduction)
                    # layer is None (default), reduction is 'mean' (default).
                    # HuBERTFeature.forward returns [L, B, H] if layer=None and reduction!="none" (actually reduction="mean" returns mean over time)
                    # Wait, HuBERTFeature code:
                    # if layer != None: ... else: out = torch.stack(out) # [L, B, T, H]
                    # if reduction == "mean": return out.mean(-2)
                    # So if layer=None, it returns [L, B, H].
                    
                    # CLaMP 3 expects MERT features. extract_clamp3.py loads .npy files.
                    # extract_mert.py saves features. If --mean_features is used (README says "averages across all layers and time steps"),
                    # README says "averages across all layers and time steps to produce a single feature per segment".
                    # extract_mert.py: if mean_features: features = features.mean(dim=0, keepdim=True)
                    # So we need to average over layers (dim 0).
                    
                    features = self.mert_model(wav_input, layer=None, reduction="mean") # [L, 1, H]
                    features = features.mean(dim=0, keepdim=True) # [1, 1, H]
                    mert_features_list.append(features)
            
            # Concatenate chunks -> (1, N_chunks, H) -> remove batch dim -> (N_chunks, H)
            # Actually CLaMP 3 expects (Batch, Seq, H) ?
            # extract_clamp3.py: input_data = np.load(filename) ... reshape(-1, input_data.size(-1))
            # It treats the whole file as a sequence of features.
            
            mert_features = torch.cat(mert_features_list, dim=0).squeeze(1) # (N_chunks, H)
            
            # Add zero vectors at start and end (from extract_clamp3.py line 122)
            zero_vec = torch.zeros((1, mert_features.size(-1)), device=self.device)
            mert_features = torch.cat((zero_vec, mert_features, zero_vec), 0)
            
            # 4. CLaMP 3 推理
            # 分段处理 (MAX_AUDIO_LENGTH)
            # extract_clamp3.py Logic:
            input_data = mert_features
            max_input_length = MAX_AUDIO_LENGTH
            
            segment_list = []
            for i in range(0, len(input_data), max_input_length):
                segment_list.append(input_data[i:i+max_input_length])
            # Handle last segment special logic in extract_clamp3.py line 131: 
            # segment_list[-1] = input_data[-max_input_length:] 
            # (This seems to imply overlap for the last segment if it's short, or just taking the last N)
            if len(segment_list) > 0:
                segment_list[-1] = input_data[-max_input_length:]
            
            last_hidden_states_list = []
            
            with torch.no_grad():
                for input_segment in segment_list:
                    # Prepare masks
                    input_masks = torch.ones(input_segment.size(0), device=self.device)
                    
                    # Pad to MAX_AUDIO_LENGTH
                    pad_len = MAX_AUDIO_LENGTH - input_segment.size(0)
                    if pad_len > 0:
                        pad_indices = torch.zeros((pad_len, AUDIO_HIDDEN_SIZE), device=self.device)
                        input_segment = torch.cat((input_segment, pad_indices), 0)
                        
                        mask_pad = torch.zeros(pad_len, device=self.device)
                        input_masks = torch.cat((input_masks, mask_pad), 0)
                    
                    # CLaMP 3 Forward
                    last_hidden_states = self.clamp_model.get_audio_features(
                        audio_inputs=input_segment.unsqueeze(0), # (1, L, H)
                        audio_masks=input_masks.unsqueeze(0),    # (1, L)
                        get_global=True
                    )
                    last_hidden_states_list.append(last_hidden_states)
            
            # Aggregation (Weighted Average)
            # extract_clamp3.py line 166
            full_chunk_cnt = len(input_data) // max_input_length
            remain_chunk_len = len(input_data) % max_input_length
            
            if remain_chunk_len == 0:
                feature_weights = torch.tensor([max_input_length] * full_chunk_cnt, device=self.device).view(-1, 1)
            else:
                feature_weights = torch.tensor([max_input_length] * full_chunk_cnt + [remain_chunk_len], device=self.device).view(-1, 1)
            
            # Ensure dimensions match
            if len(last_hidden_states_list) != feature_weights.shape[0]:
                # Fallback or simple mean if logic mismatch
                feature_weights = torch.ones((len(last_hidden_states_list), 1), device=self.device)
            
            last_hidden_states_list = torch.concat(last_hidden_states_list, 0) # (N_seg, H)
            last_hidden_states_list = last_hidden_states_list * feature_weights
            final_feature = last_hidden_states_list.sum(dim=0) / feature_weights.sum() # (H,)
            
            # 归一化
            final_feature = final_feature / final_feature.norm(dim=-1, keepdim=True)
            
            return final_feature.unsqueeze(0).cpu().numpy() # (1, D)
            
        except Exception as e:
            console.print(f"[red]音频编码失败: {audio_path} - {e}[/red]")
            import traceback
            console.print(traceback.format_exc())
            return None
        
    def retrieve_tags(
        self,
        audio_path: Path,
        top_k: int = 5
    ) -> SemanticTags:
        """
        检索最匹配的语义标签
        
        Args:
            audio_path: 音频文件路径
            top_k: 每个类别返回的标签数量
            
        Returns:
            SemanticTags 对象
        """
        if not self._loaded:
            self.load_model()
            
        console.print(f"\n[cyan]语义检索: {audio_path.name}[/cyan]")
        
        # 获取音频嵌入
        # Shape: (1, D)
        audio_embedding = self.encode_audio(audio_path)
        
        tags = SemanticTags()
        
        if audio_embedding is not None and self.descriptor_bank.embeddings is not None:
            # 文本 Embeddings: (N, D)
            text_embeddings = self.descriptor_bank.embeddings
            
            # 计算余弦相似度: (1, D) @ (D, N) -> (1, N)
            # 假设都已归一化
            similarities = (audio_embedding @ text_embeddings.T).squeeze() # (N,)
            
            # 整理结果
            # 为了按类别筛选，我们需要遍历所有结果
            # 建立一个 (score, category, tag) 的列表
            results = []
            for idx, score in enumerate(similarities):
                cat, tag = self.descriptor_bank.flat_tags[idx]
                results.append({
                    "category": cat,
                    "tag": tag,
                    "score": float(score)
                })
            
            # 按类别分组并排序
            from collections import defaultdict
            grouped_results = defaultdict(list)
            for res in results:
                grouped_results[res["category"]].append(res)
                
            # 填充 SemanticTags
            # 每个类别取 Top-K
            all_scores = {}
            
            for category, items in grouped_results.items():
                # 降序排序
                items.sort(key=lambda x: x["score"], reverse=True)
                
                # 取 Top-K (且分数需大于某个微小阈值，比如 0.05)
                top_items = [item for item in items[:top_k] if item["score"] > 0.05]
                
                tag_names = [item["tag"] for item in top_items]
                
                # 赋值给 SemanticTags 对应的字段
                if category == "mood":
                    tags.mood = tag_names
                elif category == "genre":
                    tags.genre = tag_names
                elif category == "instruments":
                    tags.instruments = tag_names
                elif category == "texture":
                    tags.texture = tag_names
                
                # 记录置信度
                for item in top_items:
                    all_scores[item["tag"]] = item["score"]
            
            tags.confidence_scores = all_scores
            
        return tags
        
    def analyze(
        self,
        audio_path: Path,
        stems_paths: Optional[Dict] = None
    ) -> SemanticTags:
        """
        完整的语义分析流程
        """
        console.print("\n[bold cyan]🎭 语义分析 (CLaMP 3)[/bold cyan]")
        
        # 分析原始音频
        main_tags = self.retrieve_tags(audio_path)
        
        return main_tags


def analyze_semantics(state: AnalysisState) -> AnalysisState:
    """
    LangGraph 节点函数：语义分析
    """
    console.print("\n[bold magenta]=== 语义评审专家 ===[/bold magenta]")
    
    start_time = time.time()
    
    try:
        analyzer = SemanticAnalyzer()
        
        # 决定分析什么：原曲
        tags = analyzer.analyze(state['audio_path'])
        
        # 更新状态
        new_state = state.copy()
        new_state['semantic_tags'] = tags
        
        if 'processing_time' not in new_state:
            new_state['processing_time'] = {}
        new_state['processing_time']["semantic_analysis"] = time.time() - start_time
        
        return new_state
        
    except Exception as e:
        console.print(f"[red][ERROR] 语义分析失败: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        
        new_state = state.copy()
        if 'errors' not in new_state:
            new_state['errors'] = []
        new_state['errors'].append(f"语义分析失败: {str(e)}")
        return new_state


# === 初始化描述符库 ===
def init_descriptor_bank():
    """初始化并保存默认描述符库"""
    bank = DescriptorBank()
    bank.save_default()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python -m src.agents.semantic_reviewer <audio_file>")
        sys.exit(1)
        
    audio_file = Path(sys.argv[1])
    
    if not audio_file.exists():
        print(f"文件不存在: {audio_file}")
        sys.exit(1)
        
    analyzer = SemanticAnalyzer()
    tags = analyzer.analyze(audio_file)
    
    print("\nTop Tags:")
    print(tags.model_dump_json(indent=2))
