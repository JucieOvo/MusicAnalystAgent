# Git 忽略规则配置实施方案 (V2)

## 1. 背景与目的
当前项目根目录下缺失 `.gitignore` 文件。根据用户指示，需要排除特定的输出目录、模型目录、第三方库目录及音频文件，确保代码仓库仅包含核心源码。

## 2. 拟定忽略内容

建议创建 `.gitignore` 文件并包含以下规则：

### 2.1 用户指定排除 (User Specified)
- `output/` (分析结果与音频分离产物)
- `models/` (排除整个模型目录，包含权重及配置文件)
- `lib/clamp3/` (排除整个 clamp3 库目录)
- `卡农.mp3` (特定音频文件，建议同时添加通用音频格式忽略规则)

### 2.2 Python 通用规则
- `__pycache__/`
- `*.pyc`
- `*.pyo`
- `*.pyd`
- `.Python`
- `env/`, `venv/`, `.env/`, `.venv/` (虚拟环境)

### 2.3 日志与调试文件
- `debug.log`
- `debug_output.txt`
- `*.log`

### 2.4 其他音频格式 (建议)
- `*.wav`
- `*.flac`
- `*.mp3` (覆盖具体的 mp3 文件)

### 2.5 系统与 IDE 配置
- `.vscode/`
- `.idea/`
- `.DS_Store`
- `Thumbs.db`

### 2.6 敏感信息
- `.env` (保留 `.env.example`)

## 3. 最终 .gitignore 文件预览
```gitignore
# User Specified
output/
models/
lib/clamp3/
卡农.mp3

# Python General
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.env/
.venv/

# Logs
debug.log
debug_output.txt
*.log

# Audio Files
*.wav
*.flac
*.mp3

# System & IDE
.vscode/
.idea/
.DS_Store
Thumbs.db

# Sensitive
.env
```

## 4. 执行计划
待批准后，将在项目根目录创建 `.gitignore` 文件并写入上述内容。
