# 🎬 video2knowledge

> 把视频变成**带时间戳的字幕 → 结构化知识文档 → HTML / Anki 卡片**。两条本地推理路径，全本地、无云端 API，每次运行留痕到公开仓库可复现。

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](#硬件适配)
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

## 🚀 快速开始

### 1. 环境要求

`ollama` + `ffmpeg` + `python3`（或 `uv`）。一键检查并下载模型：

```bash
bash scripts/setup_models.sh
```

这个脚本会**自动检测你的机型**（RAM / GPU / Apple Silicon / NVIDIA），按档位推荐并拉取对应的 VLM、装好 faster-whisper + genanki。看它给你选了什么：

```bash
python3 scripts/hardware_profile.py
# 例：8GB MacBook → profile=mid → whisper-small + minicpm-v4.6
```

### 2. 一条命令跑通（路径 2 · ASR，推荐先试）

```bash
source .venv/bin/activate

# 第一步：视频 → 字幕
python3 scripts/asr_caption.py \
  --video your_video.mp4 --out-dir runs/demo --language zh

# 第二步：字幕 → 知识文档 / HTML / 卡片 CSV
python3 scripts/build_knowledge.py \
  --subtitles runs/demo/subtitles.json --out-dir runs/demo --format all

# 2.3：CSV → Anki 牌组
python3 scripts/gen_apkg.py \
  --csv runs/demo/cards.csv --out runs/demo/cards.apkg --deck "我的知识卡"
```

完成后 `runs/demo/` 里就有 `subtitles.srt`、`knowledge.md`、`knowledge.html`、`cards.csv`、`cards.apkg`。

### 3. 路径 1 · 多模态（无音轨 / 屏幕录制）

```bash
python3 scripts/mm_caption.py \
  --video screen_recording.mp4 --out-dir runs/demo2 --interval 2.0
# 再走同样的第二步
```

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

## 🔍 留痕（可复现）

每次运行建议放进 `runs/<时间戳>-<视频名>/`，并写一份 `manifest.json` 记录视频源、路径、模型、参数、产出清单。仓库本身即审计 trail——任何人能从 `manifest.json` + skill 版本复现一次运行。

`example/` 目录里有一份完整跑通的示例产出（ffmpeg testsrc 视频的多模态路径），`runs/` 里有真实中文视频（百度网盘同步空间教程）的 ASR 路径产出，可直接打开验收。

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
├── example/                       # 跑通的示例产出
└── runs/                          # 真实视频测试产出（留痕）
```

---

## 🧠 设计取舍

- **全本地推理**：用 Ollama 跑 VLM/文本模型、faster-whisper 跑 ASR，视频内容不离开本机，隐私可控、留痕可复现。
- **小模型优先**：默认档位（mid）用 1B 级模型，8GB 机器跑得动；模型小→摘要/时间轴/知识点质量好，但 Q&A 在 1B 模型上偶有偏差。换更大的本地文本模型（`build_knowledge.py --model qwen2.5:7b`）即可显著改善。
- **CLI 优先，可被 skill 调用**：所有脚本带 `--help`、不硬编码路径、幂等。

---

## 📜 License

[MIT](LICENSE) © CacinieP
