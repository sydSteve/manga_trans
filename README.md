# Manga Translator Pipeline

[English](#english) | [中文](#chinese)

<a id="english"></a>
## English

Local manga page translation pipeline for Japanese comics. The current repository provides:

- Text region detection
- Speech bubble detection
- Japanese OCR
- Optional DeepSeek translation
- Source-region cleanup
- Chinese text rendering

This repository is currently organized around a Windows + PowerShell + NVIDIA CUDA 12.4 workflow, with the main pipeline living under `app/`.

### Repository Layout

```text
app/          Main pipeline code, config, and CLI entry
third_party/  Third-party models (not committed to GitHub by default)
data/         Input and output image directories
test/         Test / mirrored experiment directory
```

### Recommended Environment

- Windows 10/11
- Python 3.11
- NVIDIA GPU
- PyTorch 2.6.0 with CUDA 12.4

CPU-only execution is possible, but it will be much slower.

### Installation

These steps assume you just downloaded or cloned the repository from GitHub.

#### 1. Clone the repository

```powershell
git clone https://github.com/sydSteve/manga_trans
cd your folder
```

#### 2. Create a virtual environment

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

#### 3. Install PyTorch

This project is currently aligned with:

- `torch 2.6.0`
- `torchvision 0.21.0`
- `torchaudio 2.6.0`

For CUDA 12.4:

```powershell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

For CPU only:

```powershell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

#### 4. Install the remaining dependencies

```powershell
pip install -r requirements.txt
```

#### 5. Prepare runtime directories

```powershell
New-Item -ItemType Directory -Force data\input_images | Out-Null
New-Item -ItemType Directory -Force data\output_images | Out-Null
New-Item -ItemType Directory -Force debug | Out-Null
New-Item -ItemType Directory -Force third_party | Out-Null
```

#### 6. Download models

Large model files are not committed to GitHub by default. You need to place them under `third_party/`.

##### 6.1 Required model

At minimum, the OCR stage requires `manga-ocr-base`:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='kha-white/manga-ocr-base', local_dir='third_party/manga-ocr-base')"
```

Expected files:

```text
third_party/manga-ocr-base/
  config.json
  preprocessor_config.json
  pytorch_model.bin
  special_tokens_map.json
  tokenizer_config.json
  vocab.txt
```

##### 6.2 Recommended models

These are strongly recommended for good results.

Text detection:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ogkalu/comic-text-and-bubble-detector', local_dir='third_party/comic-text-and-bubble-detector')"
```

Speech bubble detection:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='mayocream/speech-bubble-segmentation', local_dir='third_party/speech-bubble-segmentation')"
```

##### 6.3 Optional model

Font-style recognition is optional. If it is missing, the pipeline falls back to heuristic style analysis.

Expected directory:

```text
third_party/yuzumarker-font-detection/
  yuzumarker-font-detection.safetensors
  font-labels.json
  font-labels-ex.json
```

#### 7. Configure environment variables

Copy the example file:

```powershell
Copy-Item app\.env.example app\.env
```

By default:

- `USE_DEEPSEEK_API=false`
- The pipeline can still run without a DeepSeek API key
- Translation will fall back to direct OCR text output

To enable real translation, edit `app/.env`:

```env
USE_DEEPSEEK_API=true
DEEPSEEK_API_KEY=your_api_key
MANGA_TRANSLATOR_DEVICE=cuda
```

### Run the Project

Put input images into:

```text
data/input_images/
```

Then run:

```powershell
cd app
python trans.py --debug
```

Or use the batch entry:

```powershell
cd app
.\trans.cmd
```

Default paths:

- Input: `data/input_images`
- Output: `data/output_images`
- Debug JSON: `data/output_images/_debug`

### Minimum Runnable Setup

If you only want to confirm that the project can run end to end, the minimum suggested setup is:

- Install Python dependencies
- Install PyTorch
- Download `third_party/manga-ocr-base`
- Keep `USE_DEEPSEEK_API=false`

In that mode:

- Text detection may fall back to EasyOCR
- Bubble detection may fall back to edge bubbles
- Translation may degrade to direct OCR text output
- Style analysis may fall back to heuristic logic

### FAQ

#### `torch.cuda.is_available()` returns `false`

Your PyTorch build likely does not match your CUDA environment, or the GPU driver is not set up correctly. Recheck step 3 first.

#### I cloned the repo but there are no model files

That is expected. The repository contains code and lightweight config only. Large models must be downloaded into `third_party/`.

#### Can I run it without a DeepSeek API key?

Yes. The project still runs, but translation will fall back to OCR text output.

#### Can I run it on Linux or macOS?

In principle yes, but `app/config.yaml` currently assumes Windows-oriented font paths. You will likely need to adjust font configuration.

### Main Pipeline References

- [app/PIPELINE_FLOW.md](app/PIPELINE_FLOW.md)
- [app/trans.py](app/trans.py)
- [app/manga_translator/cli.py](app/manga_translator/cli.py)

[Back to top](#manga-translator-pipeline)

<a id="chinese"></a>
## 中文

这是一个面向日文漫画页面的本地翻译流水线。当前仓库主要提供：

- 文本区域检测
- 气泡检测
- 日文 OCR
- 可选的 DeepSeek 翻译
- 原文区域修复
- 中文嵌字输出

当前仓库默认按 `Windows + PowerShell + NVIDIA CUDA 12.4` 环境整理，主流程位于 `app/` 目录。

### 仓库结构

```text
app/          主流程代码、配置、CLI 入口
third_party/  第三方模型目录（默认不提交到 GitHub）
data/         输入输出图片目录
test/         测试 / 镜像实验目录
```

### 推荐环境

- Windows 10/11
- Python 3.11
- NVIDIA GPU
- 对应 CUDA 12.4 的 PyTorch 2.6.0

也可以使用 CPU 跑通，但速度会明显更慢。

### 安装步骤

以下步骤从“刚从 GitHub 下载仓库代码”开始。

#### 1. 克隆仓库

```powershell
git clone https://github.com/sydSteve/manga_trans
cd 你下载的文件路径
```

#### 2. 创建虚拟环境

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

#### 3. 安装 PyTorch

当前项目对应的版本是：

- `torch 2.6.0`
- `torchvision 0.21.0`
- `torchaudio 2.6.0`

如果使用 CUDA 12.4：

```powershell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

如果只使用 CPU：

```powershell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

#### 4. 安装其余依赖

```powershell
pip install -r requirements.txt
```

#### 5. 准备运行目录

```powershell
New-Item -ItemType Directory -Force data\input_images | Out-Null
New-Item -ItemType Directory -Force data\output_images | Out-Null
New-Item -ItemType Directory -Force debug | Out-Null
New-Item -ItemType Directory -Force third_party | Out-Null
```

#### 6. 下载模型

由于 GitHub 不适合直接托管大模型文件，`third_party/` 下的模型需要手动下载。

##### 6.1 必需模型

最少需要安装 `manga-ocr-base` 才能完成 OCR：

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='kha-white/manga-ocr-base', local_dir='third_party/manga-ocr-base')"
```

目录中至少应包含：

```text
third_party/manga-ocr-base/
  config.json
  preprocessor_config.json
  pytorch_model.bin
  special_tokens_map.json
  tokenizer_config.json
  vocab.txt
```

##### 6.2 推荐模型

以下模型强烈建议安装，否则效果会退化到回退逻辑。

文本检测模型：

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ogkalu/comic-text-and-bubble-detector', local_dir='third_party/comic-text-and-bubble-detector')"
```

气泡检测模型：

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='mayocream/speech-bubble-segmentation', local_dir='third_party/speech-bubble-segmentation')"
```

##### 6.3 可选模型

字体风格识别模型是可选项。如果缺失，流程会回退到启发式样式分析，不会阻塞主流程。

期望目录如下：

```text
third_party/yuzumarker-font-detection/
  yuzumarker-font-detection.safetensors
  font-labels.json
  font-labels-ex.json
```

#### 7. 配置环境变量

复制示例文件：

```powershell
Copy-Item app\.env.example app\.env
```

默认情况下：

- `USE_DEEPSEEK_API=false`
- 没有 DeepSeek API Key 也可以运行
- 翻译阶段会退化为直接输出 OCR 文本

如果要启用真实翻译，请编辑 `app/.env`：

```env
USE_DEEPSEEK_API=true
DEEPSEEK_API_KEY=your_api_key
MANGA_TRANSLATOR_DEVICE=cuda
```

### 运行项目

把输入图片放到：

```text
data/input_images/
```

然后执行：

```powershell
cd app
python trans.py --debug
```

也可以使用批处理入口：

```powershell
cd app
.\trans.cmd
```

默认路径：

- 输入：`data/input_images`
- 输出：`data/output_images`
- 调试 JSON：`data/output_images/_debug`

### 最低可运行配置

如果你只想先验证项目能不能跑通，最低建议是：

- 安装 Python 依赖
- 安装 PyTorch
- 下载 `third_party/manga-ocr-base`
- 保持 `USE_DEEPSEEK_API=false`

在这种模式下：

- 文本检测可能回退到 EasyOCR
- 气泡检测可能回退到 edge bubbles
- 翻译可能退化为 OCR 文本直出
- 风格分析可能退化为启发式逻辑

### 常见问题

#### `torch.cuda.is_available()` 返回 `false`

通常说明 PyTorch 与你的 CUDA 环境不匹配，或者显卡驱动没有正确配置。优先重新检查第 3 步。

#### 克隆仓库后没有模型文件

这是正常的。仓库默认只放代码和轻量配置，大模型需要单独下载到 `third_party/`。

#### 没有 DeepSeek API Key 能不能运行

可以。只是翻译会退化为 OCR 文本输出。

#### Linux / macOS 能不能跑

原则上可以，但 `app/config.yaml` 当前默认字体路径偏向 Windows，Linux / macOS 上通常需要自行调整字体配置。

### 主流程参考

- [app/PIPELINE_FLOW.md](app/PIPELINE_FLOW.md)
- [app/trans.py](app/trans.py)
- [app/manga_translator/cli.py](app/manga_translator/cli.py)

[返回顶部](#manga-translator-pipeline)
