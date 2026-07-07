# 🎬 video2knowledge

> 把视频变成**带时间戳的字幕 → 结构化知识文档 → HTML / Anki 卡片**。两条本地推理路径，**全程在本地运行，不上传任何视频/字幕/产出**；仓库只跟踪代码与配置变更。

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](#-从零开始安装5-分钟)
[![Models](https://img.shields.io/badge/models-local-faster--whisper%20%2F%20ollama-green.svg)](#两条路径)

---

## ✨ 它能做什么

给它任意一段视频，你会得到：

| 产物 | 文件 | 说明 |
|---|---|---|
| 📝 **带时间戳字幕** | `subtitles.srt` / `.vtt` / `.json` | 词级时间戳，可直接喂播放器或下游处理 |
| 📄 **知识文档** | `knowledge.md` | 摘要 / 时间轴 / 核心知识点 / Q&A / 术语表，**支持自定义模板** |
| 🌐 **HTML** | `knowledge.html` | 自包含单文件，`[mm:ss]` 时间戳可点跳 |
| 🃏 **知识卡片 CSV** | `cards.csv` | question / answer / tags / timestamp / source |
| 📚 **Anki 牌组** | `cards.apkg` | 稳定 ID，重复导入不重复，开箱即用 |

**两条路径**任选或并用：

- **路径 1 · 多模态**：原生多模态小模型（≤4B VLM，经 Ollama）逐帧读视频 → 带时间戳字幕。适合**无音轨 / 纯画面 / 屏幕录制 / 演示文稿**，能抓 ASR 看不见的屏幕文字和图表。
- **路径 2 · ASR**：faster-whisper 转写音轨 → 带时间戳字幕。适合**有清晰语音的视频**（讲座/访谈/教程），更快更准。

两条路径产出的字幕 schema 一致，第二步（知识加工）对路径无感。

---

## 🧱 从零开始安装（5 分钟）

假设你是一台**干净的系统**（没装 ollama / ffmpeg / python），下面四步就能从 0 跑通。

### 第 0 步 · 克隆仓库

```bash
git clone https://github.com/CacinieP/video2knowledge.git
cd video2knowledge
```

### 第 1 步 · 安装系统依赖（三件套）

需要三个命令行工具，按你的系统挑一组：

**macOS（用 [Homebrew](https://brew.sh)）**
```bash
brew install ffmpeg python@3.11          # ffmpeg + python
brew install ollama                       # 或去 https://ollama.com/download 下 Ollama.app
```

**Linux（apt，Debian/Ubuntu）**
```bash
curl -fsSL https://ollama.com/install.sh | sh          # ollama 官方脚本
sudo apt update && sudo apt install -y ffmpeg python3 python3-venv
```

**Windows**
- Ollama：<https://ollama.com/download> 下载安装包
- ffmpeg / python：`winget install Gyan.FFmpeg Python.Python.3.11`
- 建议在 **Git Bash** 或 **WSL** 里运行下面的命令

> **检查**：`ollama --version && ffmpeg -version && python3 --version` 三条都有输出即可继续。

### 第 2 步 · 一键下载模型 + 建虚拟环境

```bash
bash scripts/setup_models.sh
```

这个脚本会做三件事（**幂等，可重复执行**）：

1. **自动检测你的机型**（RAM / GPU / Apple Silicon / NVIDIA），按档位挑模型；
2. 启动 `ollama serve` 并拉取对应的 **VLM**（多模态，路径 1 用）；
3. 在 `~/.zcode/skills/video2knowledge/.venv` 建一个 venv，装好 `faster-whisper` + `genanki`。

跑完会打印一段总结，**注意看最后一行 `Activate with:`**，那是要复制的激活命令。

看看它给你选了什么档位：

```bash
python3 scripts/hardware_profile.py
# 例：8GB MacBook → profile=mid → whisper-small + minicpm-v4.6
```

> 💡 **下载慢 / 卡住？** 这些模型从 Ollama / PyPI 拉取，国内网络可设置代理提速：
> ```bash
> export HTTPS_PROXY=http://127.0.0.1:7890
> bash scripts/setup_models.sh
> ```

### 第 3 步 · 激活 venv（每次新开终端都要做）

```bash
# 路径以 setup_models.sh 末尾打印的为准（默认如下）：
source ~/.zcode/skills/video2knowledge/.venv/bin/activate
```

> 想让 venv 直接建在仓库内？跑 setup 时覆盖一下环境变量即可：
> `VENV_DIR=.venv bash scripts/setup_models.sh`，之后用 `source .venv/bin/activate`。

到这里环境就装好了。下面正式处理视频。

---

## 🚀 处理第一个视频

### A. 路径 2 · ASR（有语音的视频，推荐先试）

```bash
source ~/.zcode/skills/video2knowledge/.venv/bin/activate   # 激活 venv

# 第一步：视频 → 字幕
python3 scripts/asr_caption.py \
  --video your_video.mp4 --out-dir runs/demo --language zh

# 第二步：字幕 → 知识文档 / HTML / 卡片 CSV
# （首次运行会自动拉取文本模型 openbmb/minicpm5:Q4_K_M，几百 MB，稍等）
python3 scripts/build_knowledge.py \
  --subtitles runs/demo/subtitles.json --out-dir runs/demo --format all

# 2.3：CSV → Anki 牌组
python3 scripts/gen_apkg.py \
  --csv runs/demo/cards.csv --out runs/demo/cards.apkg --deck "我的知识卡"
```

完成后 `runs/demo/` 里就有 `subtitles.srt`、`knowledge.md`、`knowledge.html`、`cards.csv`、`cards.apkg`。

> 英文视频记得在第二步加 `--lang en`（默认 `zh`），否则小模型在语言不匹配时容易把示例内容串进产出。

### B. 路径 1 · 多模态（无音轨 / 屏幕录制 / 演示文稿）

```bash
python3 scripts/mm_caption.py \
  --video screen_recording.mp4 --out-dir runs/demo2 --interval 2.0
# 再走同样的第二步（build_knowledge.py），输出与路径 2 完全一致
```

---

## 🎯 实际效果演示

下面是一段 **NASA 公有领域视频**（Curiosity 火星车着陆后 Adam Steltzner 的发言，2分25秒，英文，[来源](https://commons.wikimedia.org/wiki/File:Curiosity_Rover_Begins_Mars_Mission_August_6_2012_-_Adam_Steltzner_speech.webm)，Public Domain）经过完整流水线后的真实产出。

**输入字幕（`subtitles.srt`，faster-whisper small 模型，19 段，前 3 段）：**
```
1
00:00:02,060 --> 00:00:03,580
Say something profound.

2
00:00:06,540 --> 00:00:09,280
I am terribly humbled by this experience.

3
00:00:11,840 --> 00:00:20,600
I forever secretly have felt that I do not deserve to be in the
position of leading the...
```

**生成的知识文档摘要（`knowledge.md`）：**
> The video explores the profound humility felt by a scientist who acknowledges
> his own limitations while recognizing the immense value of working with a
> diverse team at JPL, highlighting how collective effort and individual
> contributions can achieve great things together... underscoring the importance
> of appreciating both the small details of daily tasks and the larger
> achievements achieved through unity.

**核心知识点（自动提炼）：**
- Leading requires recognizing individual contributions.
- Team success depends on diverse skills and perspectives.
- Humility is essential for learning from others.
- Every great achievement involves collaboration.

**知识卡片（`cards.csv` → `cards.apkg`，可直接导入 Anki）：**

| Question | Answer |
|---|---|
| How does the speaker feel about leading a team? | Expresses humility — "secretly have felt that I do not deserve to be in the position of leading." |
| What is the significance of the EDL team? | Described as talent at JPL, emphasizing collective skill and mission contribution. |
| Why does the speaker believe this nation represents humanity? | A "corner of humanity that reaches out and explores," highlighting its role in exploration. |

> 💡 **提示**：英文视频请加 `--lang en`（中文视频用默认 `--lang zh`）。小模型（1B）若语言不匹配会把示例内容串进产出。

---

## 🖥️ 硬件适配（自动）

不用手动挑模型大小——`scripts/hardware_profile.py` 会检测并匹配：

| Profile | 触发 | ASR 模型 | VLM | 典型机型 |
|---|---|---|---|---|
| `tiny` | RAM < 6 GB | tiny | moondream | 树莓派 / 4G 老笔记本 |
| `low` | 6–8 GB 无独显 | base | minicpm-v4.6 | 上网本 |
| `low-mac` | 6–8 GB Apple Silicon | small | minicpm-v4.6 | M1 MacBook Air |
| `mid` | 8–16 GB | small | minicpm-v4.6 | **主流笔记本** |
| `high` | 16–32 GB | medium | qwen2.5vl:3b | M2/M3 Pro、16G PC |
| `high-gpu` | NVIDIA ≥ 8 GB 显存 | large-v3 | qwen2.5vl:7b | RTX 3060/4060/3090（CUDA+float16 全速）|
| `max` | RAM > 32 GB | large-v3 | qwen2.5vl:7b | 工作站 / 服务器 |

NVIDIA 有短路逻辑：≥8GB 显存直接走 CUDA，不受总内存限制。全部可用环境变量（`ASR_DEFAULT_MODEL=`、`VLM_MODEL=`）或 CLI flag 覆盖。完整说明见 [`references/hardware-profiles.md`](references/hardware-profiles.md)。

---

## 📐 自定义知识文档模板

内置默认模板（`assets/default-template.md`）用 `{{占位符}}` 渲染。写任意 `.md` 放进你想要的占位符即可：

```markdown
# {{title}} — 课程笔记
> {{date}} · {{duration}} · {{source}}

## 本节目标
{{summary}}

## 时间轴
{{timeline}}

## 必背知识点
{{key_points}}

## 自测题
{{qa}}
```

可用占位符：`{{title}}` `{{source}}` `{{duration}}` `{{date}}` `{{summary}}`
`{{timeline}}` `{{key_points}}` `{{qa}}` `{{glossary}}` `{{meta}}`。
内置课程笔记 / 会议纪要 / 技术教程三套示例见 [`references/templates.md`](references/templates.md)。

```bash
python3 scripts/build_knowledge.py \
  --subtitles runs/demo/subtitles.json --out-dir runs/demo \
  --template ./my-lecture-template.md --format knowledge
```

---

## ❓ 常见问题（从零开始最容易踩的坑）

<details>
<summary><b>Q: 激活 venv 的命令到底是什么路径？</b></summary>

默认是 `~/.zcode/skills/video2knowledge/.venv`，但**以 `setup_models.sh` 末尾打印的那行为准**。也可以用 `VENV_DIR=.venv bash scripts/setup_models.sh` 把 venv 建在仓库内，之后 `source .venv/bin/activate` 即可。
</details>

<details>
<summary><b>Q: 跑 <code>build_knowledge.py</code> 卡很久 / 报模型找不到？</b></summary>

第二步会调用一个**文本模型** `openbmb/minicpm5:Q4_K_M`（用于摘要/知识点/Q&A），首次运行时 Ollama 会自动拉取，几百 MB，需要联网和等待。提前手动拉可避免等待意外：`ollama pull openbmb/minicpm5:Q4_K_M`。想换更大的模型提升质量：`--model qwen2.5:7b`。
</details>

<details>
<summary><b>Q: 报错 <code>ollama not found</code> / <code>ffmpeg not found</code>？</b></summary>

回【第 1 步】把对应工具装上并确认在 PATH 里：`ollama --version && ffmpeg -version`。Ollama 装好后若未常驻，`setup_models.sh` 会自动 `ollama serve` 拉起；若仍失败，手动开一个终端跑 `ollama serve`。
</details>

<details>
<summary><b>Q: 模型 / pip 下载很慢或超时？</b></summary>

国内网络建议挂代理：`export HTTPS_PROXY=http://127.0.0.1:7890`（端口换成你自己的）。pip 可换镜像：`pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`。
</details>

<details>
<summary><b>Q: Windows 上能跑吗？</b></summary>

可以，建议在 **Git Bash** 或 **WSL** 里运行（脚本依赖 bash）。Ollama 用官方安装包，ffmpeg/python 用 `winget` 安装，venv 激活路径同样以 `setup_models.sh` 输出为准。
</details>

---

## 🔒 隐私与留痕

- **全程本地**：视频文件、抽帧、字幕、知识产物始终留在你机器上的 `runs/<时间戳>-<视频名>/`，绝不上传，不联网调用云 API。
- **仓库只跟代码**：本仓库是 skill 本身（脚本/文档/模板）的版本管理，**不包含任何视频或处理产出**——`runs/` 已在 `.gitignore` 中忽略。代码与配置的修改都有 git 历史可追溯。
- **本地复现**：要复现某次结果，在本地 `runs/<...>/` 里查看当次用的参数和产出即可（按需自行写 `manifest.json` 记录，但默认不入库）。

`example/` 目录提供一份用 ffmpeg 合成视频跑通的示例产出（无真实数据，仅供演示结构与字段）。

---

## 📂 项目结构

```
video2knowledge/
├── SKILL.md                       # 主控文档（流程编排 + 留痕规范）
├── scripts/
│   ├── hardware_profile.py        # 机型检测 → 配置档（单一真相源）
│   ├── setup_models.sh            # 幂等：检测机型 + 拉模型 + 建 venv
│   ├── asr_caption.py             # 路径 2：faster-whisper → 字幕
│   ├── mm_caption.py              # 路径 1：VLM 逐帧 → 字幕
│   ├── extract_frames.py          # ffmpeg 抽帧 → frames.json
│   ├── build_knowledge.py         # 第二步：字幕 → 知识文档/HTML/CSV
│   └── gen_apkg.py                # 2.3：CSV → Anki .apkg
├── references/                    # 详细文档（按需加载）
│   ├── hardware-profiles.md
│   ├── path1-multimodal.md
│   ├── path2-asr.md
│   ├── templates.md
│   └── outputs.md
├── assets/default-template.md     # 内置默认知识文档模板
└── example/                       # 合成视频的示例产出（无真实数据）
```

> 处理真实视频时，产出会写到本地 `runs/`（已 gitignore，不入库）。

---

## 🧠 设计取舍

- **全本地推理**：用 Ollama 跑 VLM/文本模型、faster-whisper 跑 ASR，视频内容不离开本机，隐私可控、留痕可复现。
- **小模型优先**：默认档位（mid）用 1B 级模型，8GB 机器跑得动；模型小→摘要/时间轴/知识点质量好，但 Q&A 在 1B 模型上偶有偏差。换更大的本地文本模型（`build_knowledge.py --model qwen2.5:7b`）即可显著改善。
- **CLI 优先，可被 skill 调用**：所有脚本带 `--help`、不硬编码路径、幂等。

---

## 📜 License

[MIT](LICENSE) © CacinieP
