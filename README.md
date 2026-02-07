# 🎵 Poly-Muse Analyst

> **多模态音乐分析智能体** - 基于 LangGraph 的 AI 音乐分析系统

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 项目简介

Poly-Muse Analyst 是一个多专家智能体系统，能够将非结构化的音乐音频转化为结构化的深度分析报告。系统通过调度 **SOTA 信号处理模型** 与 **多模态理解模型**，实现对音乐音频的全维解析。

### ✨ 核心特性

| 分析层 | 能力 | 技术 |
|--------|------|------|
| 🎧 **听觉分离层** | 将混音分离为独立乐器轨道 | BS-RoFormer (SDR ~12.9dB) |
| 🎼 **符号转录层** | 将音频转换为 MIDI 数据 | Basic Pitch |
| 🎭 **语义检索层** | 识别风格、情感、音色特征 | CLaMP 3 |
| 📝 **认知综合层** | 生成专业音乐分析报告 | DeepSeek-Reasoner |

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
cd MusicAnalystAgent

# 创建虚拟环境 (推荐)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# 复制环境变量模板
copy .env.example .env

# 编辑 .env 文件，填入您的 DeepSeek API Key
```

### 3. 运行分析

```bash
# 分析音频文件
python -m src.main analyze 卡农.mp3

# 查看帮助
python -m src.main --help
```

## 📁 项目结构

```
MusicAnalystAgent/
├── src/
│   ├── __init__.py
│   ├── config.py           # 全局配置
│   ├── schemas.py          # 数据结构定义
│   ├── workflow.py         # LangGraph 工作流编排
│   ├── main.py             # CLI 入口
│   └── agents/
│       ├── separator.py    # 听觉分离专家
│       ├── transcriber.py  # 符号转录专家
│       ├── semantic_reviewer.py  # 语义理解专家
│       └── analyst.py      # 认知综合层
├── models/                  # 模型权重
│   └── model_bs_roformer_*.ckpt
├── data/
│   └── descriptor_bank.json # 语义描述符库
├── output/                  # 分析结果输出
├── docs/                    # 文档
├── requirements.txt
├── .env.example
└── README.md
```

## 🔧 命令行工具

```bash
# 完整分析
python -m src.main analyze <audio_file>

# 仅音源分离
python -m src.main separate <audio_file>

# 仅符号转录
python -m src.main transcribe <stem_file> --type vocals

# 初始化描述符库
python -m src.main init-descriptors

# 查看系统配置
python -m src.main info
```

## 📊 输出示例

分析完成后，会在 `output/<音频名>/` 目录生成：

- `stems/` - 分离的各乐器轨道
  - `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`
- `midi/` - 转录的 MIDI 文件
- `analysis_result.json` - 结构化分析数据
- `analysis_report.md` - Markdown 格式的分析报告

## 🛠️ 开发路线图

### Phase 1: MVP ✅
- [x] 项目基础架构
- [x] 配置管理系统
- [x] 数据结构定义
- [x] Agent 框架搭建
- [x] CLI 工具

### Phase 2: Agent 封装
- [ ] BS-RoFormer 集成
- [ ] Basic Pitch 集成
- [ ] CLaMP 3 集成
- [ ] 描述符库向量索引

### Phase 3: 集成优化
- [ ] LLM Prompt 优化
- [ ] FP16 推理加速
- [ ] Streamlit 前端界面

## 📚 技术文档

详细的架构设计请参阅：[音乐分析智能体架构设计文档.md](音乐分析智能体架构设计文档.md)

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

---

<p align="center">
  <b>🎵 让 AI 听懂音乐</b>
</p>
