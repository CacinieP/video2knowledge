# 🎬 video2knowledge

> **一个 Agent Skill**：把视频变成**带时间戳的字幕 → 结构化知识文档 → 图文笔记 → HTML / Anki 卡片**。
> 装进 agent 的 skills 目录后，一句自然语言（"把这节课做成笔记"）即可驱动全流程；也保留了完整的手动 CLI 流水线（见下方[古法 · 手动 CLI](#-古法--手动跑流水线cli)）。
> 三条本地推理路径，**全程本地运行，不上传任何视频/字幕/产出**；仓库只跟踪代码与配置变更。

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](#install-from-scratch)
[![Models](https://img.shields.io/badge/models-local-faster--whisper%20%2F%20ollama-green.svg)](#paths)

---

## ✨ 它能做什么

给它任意一段视频，你会得到：

| 产物 | 文件 | 说明 |
|---|---|---|
| 📝 **带时间戳字幕** | `subtitles.srt` / `.vtt` / `.json` | 词级时间戳，可直接喂播放器或下游处理 |
| 📄 **知识文档** | `knowledge.md` | 摘要 / 时间轴 / 核心知识点 / Q&A / 术语表 / **画面要点**，**支持自定义模板** |
| 🖼️ **图文笔记** | `notes.md` + `notes.html` | 关键帧插图 × 对应旁白要点交错排版；HTML 版自包含单文件可直接分享 |
| 🍵 **纯享版笔记** | `notes-distilled.md` | 同样的节点和插图，砍掉原声引用——背起来/复习用，体积少 42%，**零额外模型开销** |
| 🌐 **HTML** | `knowledge.html` | 自包含单文件，`[mm:ss]` 时间戳可点跳 |
| 🃏 **知识卡片 CSV** | `cards.csv` | question / answer / tags / timestamp / source，**自动去重** |
| 📚 **Anki 牌组** | `cards.apkg` | 稳定 ID，重复导入不重复，开箱即用 |

<a id="paths"></a>

**三条路径**任选或并用：

- **路径 1 · 多模态**：原生多模态小模型（≤4B VLM，经 Ollama）逐帧读视频 → 带时间戳字幕。适合**无音轨 / 纯画面 / 屏幕录制 / 演示文稿**，能抓 ASR 看不见的屏幕文字和图表。
- **路径 2 · ASR**：把音轨转写成带时间戳字幕。适合**有清晰语音的视频**（讲座/访谈/教程），更快更准。中文推荐 **FunASR Paraformer**：同一条 30 分钟课、同样的 39 条热词，领域同音词正确率从 faster-whisper 的 **44% → 94%**，而且跑 CPU，12–16× 实时；`python3 scripts/hardware_profile.py --recommend` 会按你的机器给出建议（**按内存判断，不是按显存**——有显卡反而要留给 VLM，ASR 和 VLM 抢显存会把 8-token 的请求从 3.8 秒拖到 60 秒以上）。转写完再跑一遍 `fix_homophones.py` 修领域同音错字。
- **路径 3 · 音画融合**：ASR 抓讲解 + VLM OCR 抓屏幕（表格/公式/举例），按时间戳融合，外加**跨路径反馈**——OCR 术语回灌 ASR 热词、语义对齐校正讲解/幻灯片错位。适合**有语音讲解的 PPT/幻灯片视频**——把 ASR 听不到的画面内容补回来。

路径 1、2 产出的字幕 schema 一致，第二步（知识加工）对路径无感；路径 3 产出融合的 `merged.json`，由第二步的 `--merged` 消费。

---

## 🤖 作为 Agent Skill 使用（推荐方式）

本仓库首先是一个 **skill**：`SKILL.md` 是给 agent 读的"操作手册"（路径选择决策树、参数推荐、产物规范、断点续跑约定），装好后 agent 会自动加载它并编排整条流水线——**你只需要说人话**：

```text
"用 video2knowledge 把这节课做成笔记"          → agent 自动选路径 3：ASR + OCR + 融合 + 全套产出
"这个录屏没声音，帮我整理一下"                 → agent 自动选路径 1（多模态）
"把这批课程视频全部跑一遍，刘忠的优先"          → agent 调 batch_run.py 批量流水线
"笔记里每条要点要带上幻灯片内容"               → agent 用 {{bullets}} 音画合并要点
```

把整个仓库 clone 进对应 agent 的 skills 目录即可（任选其一，不互斥）：

| Agent | 安装目录 | 安装命令 |
|---|---|---|
| **ZCode** | `~/.zcode/skills/video2knowledge/` | `git clone https://github.com/CacinieP/video2knowledge.git ~/.zcode/skills/video2knowledge` |
| **Claude Code** | `~/.claude/skills/video2knowledge/` | `git clone https://github.com/CacinieP/video2knowledge.git ~/.claude/skills/video2knowledge` |
| **Cursor** | `~/.cursor/skills/video2knowledge/` | `git clone https://github.com/CacinieP/video2knowledge.git ~/.cursor/skills/video2knowledge` |

首次使用让 agent 跑一次 `bash scripts/setup_models.sh`（或你自己跑），它会自动探测机型、拉取适配模型、建好 `.venv/`——之后一切交给对话。

> 不用 agent、只想在终端里跑脚本？往下看安装步骤和[古法 · 手动 CLI](#-古法--手动跑流水线cli)。

---

## 🧱 从零开始安装（5 分钟）

<a id="install-from-scratch"></a>

假设你是一台**干净的系统**（没装 ollama / ffmpeg / python），下面四步就能从 0 跑通。

### 第 0 步 · 克隆仓库

打算作为 skill 用（推荐）就直接 clone 进 skills 目录（见上文表格，一次到位）；只想试试 CLI 则任意目录：

```bash
# skill 用法（以 ZCode 为例，一次到位）
git clone https://github.com/CacinieP/video2knowledge.git ~/.zcode/skills/video2knowledge
cd ~/.zcode/skills/video2knowledge

# 或：仅 CLI 试玩
git clone https://github.com/CacinieP/video2knowledge.git && cd video2knowledge
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

1. **自动检测你的机型**（RAM / GPU / Apple Silicon / NVIDIA，跨平台探测链），按档位挑模型；
2. 启动 `ollama serve` 并拉取对应的 **VLM**（多模态，路径 1 用）；
3. 在**仓库根目录**建一个 `.venv`，装好 `faster-whisper` + `genanki`。

跑完会打印一段总结，**注意 `run python as:` 那一行**——那是本机 venv 解释器的绝对路径（Windows 是 `.venv/Scripts/python.exe`，macOS/Linux 是 `.venv/bin/python`，脚本自动识别）。

想建在别处？用环境变量 `VENV_DIR=...` 覆盖。

看看它给你选了什么档位：

```bash
python3 scripts/hardware_profile.py
# 例：8GB MacBook → profile=mid → whisper-small + minicpm-v4.6 + minicpm5-2b
# 例：31GB Win11 台式机 → profile=high → whisper-medium + qwen3.5:4b（视觉+文本共用一次拉取）
```

> 💡 **下载慢 / 卡住？** 这些模型从 Ollama / PyPI 拉取，国内网络可设置代理提速：
> ```bash
> export HTTPS_PROXY=http://127.0.0.1:7890
> bash scripts/setup_models.sh
> ```

### 第 3 步 · 运行时用哪个 python

示例里统一写 `python3`（已激活 venv 的前提下）。两种等价用法任选：

```bash
# 用法 A：激活 venv（macOS/Linux；Windows Git Bash 用 Scripts/activate）
source .venv/bin/activate            # 或 .venv/Scripts/activate

# 用法 B：不激活，直接用绝对路径（就是 setup_models.sh 打印的 run python as）
.venv/bin/python scripts/asr_caption.py ...      # macOS/Linux
.venv/Scripts/python.exe scripts/asr_caption.py ...  # Windows
```

### 第 4 步 · 接回 Agent（如果第 1 步没装）

如果第 1 步装在 skills 目录，这一步什么都不用做；如果当时 clone 到了普通目录，把它挪进上文任一 skills 目录即可。仓库内置的 `scripts/setup_models.sh` 会把 venv 建在**仓库根目录的 `.venv/`**，装到任何 agent 目录都一样，无需额外配置。

到这里环境就装好了。下面是手动跑流水线的方式（古法）。

---

## ⌨️ 古法 · 手动跑流水线（CLI）

> 命令行直接驱动各脚本，适合调试、CI、或不想开 agent 的场合。日常使用建议直接对 agent 说需求，它会替你拼这些命令。

### A. 路径 2 · ASR（有语音的视频，推荐先试）

先问机器要建议（中文会推荐 FunASR Paraformer）：

```bash
python3 scripts/hardware_profile.py --recommend
```

```bash
source .venv/bin/activate   # 激活 venv（用了方式 B 则换成 setup_models.sh 打印的那行）

# 第一步：视频 → 字幕
python3 scripts/asr_caption.py \
  --video your_video.mp4 --out-dir runs/demo --language zh

#   中文更准的选择：FunASR Paraformer（领域同音词 44% → 94%，跑 CPU，12-16× 实时）
#   需要独立 venv —— funasr 的 tokenizers 依赖装不进主环境
python3 .venv-funasr/bin/python scripts/asr_funasr.py \
  --video your_video.mp4 --out-dir runs/demo --language zh \
  --hotwords "背谱,视谱,音阶,琶音"

# 第一步半：修领域同音错字（ASR 听的是声音不是词，错字会一路传到所有下游产物）
python3 scripts/fix_homophones.py --subtitles runs/demo/subtitles.json --dry-run  # 先看看会改什么
python3 scripts/fix_homophones.py --subtitles runs/demo/subtitles.json            # 确认后真改

# 第二步：字幕 → 知识文档 / HTML / 卡片 CSV
# （首次运行会自动拉取文本模型：mid 及以下档为 openbmb/minicpm5-2b，约 1.6 GB，稍等）
python3 scripts/build_knowledge.py \
  --subtitles runs/demo/subtitles.json --out-dir runs/demo --format all

# 2.3：CSV → Anki 牌组
python3 scripts/gen_apkg.py \
  --csv runs/demo/cards.csv --out runs/demo/cards.apkg --deck "我的知识卡"
```

完成后 `runs/demo/` 里就有 `subtitles.srt`、`knowledge.md`、`knowledge.html`、`cards.csv`、`cards.apkg`，装了 python-docx/fpdf2 时 `--format all` 还会自动产出 `knowledge.docx` 和 `knowledge.pdf`。同声修复会一并重写 `subtitles.srt` / `.vtt`，并把改动日志写进 `homophone_fixes.json`。

> 英文视频记得在第二步加 `--lang en`（默认 `zh`），否则小模型在语言不匹配时容易把示例内容串进产出。
>
> **没有语音的视频**（纯音乐、纯幻灯片）ASR 会返回 0 段。旧版照样生成"知识文档"，内容全是模型编的；现在 `knowledge_doc_status()` 会返回 `no-speech` 并拒绝伪造。

### B. 路径 1 · 多模态（无音轨 / 屏幕录制 / 演示文稿）

```bash
python3 scripts/mm_caption.py \
  --video screen_recording.mp4 --out-dir runs/demo2 --interval 2.0
# 再走同样的第二步（build_knowledge.py），输出与路径 2 完全一致
```

对 PPT/幻灯片视频，用 `--mode dedup`（通用感知去重：密采样 + dHash、稳定帧选取、黑帧过滤、簇式帧预算，无需调场景阈值）+ `--prompt-ocr`（表格/公式全量转写，自带 OCR 文本变化门控——画面变了但文字没变的帧会被丢弃）：

```bash
python3 scripts/mm_caption.py \
  --video slides.mp4 --out-dir runs/demo2 --mode dedup --prompt-ocr
# 可选：--hash-mode dual 抓纯色/渐变类画面变化（阈值约×2，如 20）；--hash-size 16 提高密集幻灯片灵敏度
```

### C. 路径 3 · 音画融合（有讲解的 PPT/幻灯片视频）

最适合线上课程、培训录屏这类「**嘴在讲、屏上有表**」的视频。ASR 抓讲解，VLM 抓屏幕上的表格/公式/举例，按时间戳融合：

```bash
source .venv/bin/activate
RUN=runs/$(date +%Y%m%d-%HMMSS)-slides; mkdir -p "$RUN"

# 1a. ASR 抓讲解（--hotwords 可注入领域术语，显著减少专有名词转写错误）
python3 scripts/asr_caption.py --video slides.mp4 --out-dir "$RUN" --language zh \
  --hotwords "亥姆霍兹自由能, 格林函数"
# 1b/1c. 感知去重抽帧 + VLM OCR 抓屏幕
python3 scripts/mm_caption.py --video slides.mp4 --out-dir "$RUN" --mode dedup --prompt-ocr
# 1d. 跨路径反馈：从 OCR 提取屏幕术语 → 热词表，并检查哪些术语 ASR 没听到
#     （覆盖率低 = ASR 听错了行话，用 --hotwords @"$RUN/ocr_hotwords.txt" 重跑 1a）
python3 scripts/hotwords_from_ocr.py \
  --captions "$RUN/captions.json" --subtitles "$RUN/subtitles.json" \
  --out "$RUN/ocr_hotwords.txt"
# 2. 按时间戳融合（+ 语义对齐校正：讲述与所附幻灯片零重叠而相邻页明显
#    更匹配时保守换绑，其余错位只标记不猜测）
python3 scripts/merge_visual.py \
  --subtitles "$RUN/subtitles.json" --visual "$RUN/captions.json" --out "$RUN/merged.json"
# 3. 生成知识文档（带"画面要点"小节；换绑/弱归属以 ⚠️ 标注给模型）
python3 scripts/build_knowledge.py \
  --subtitles "$RUN/subtitles.json" --merged "$RUN/merged.json" \
  --out-dir "$RUN" --format all
python3 scripts/gen_apkg.py --csv "$RUN/cards.csv" --out "$RUN/cards.apkg" --deck "幻灯片知识卡"
```

默认文本模型跟随 `hardware_profile.py` 的档位（`low`/`low-mac`/`mid` 档为 **openbmb/minicpm5-2b**，2.5B dense、约 1.6 GB、131K 上下文；`high` 及以上直接复用路径 1 的 VLM 权重，视觉+文本共用一次拉取）。`tiny` 档降到 `openbmb/minicpm5:Q4_K_M`（688 MB）。覆盖方式：`--model`，或 `V2K_TEXT_MODEL=` 全局固定。详见 `references/path3-fusion.md`。

### D. 图文笔记（关键帧插图 × 旁白要点）

把去重关键帧和对应时间段的旁白交错排版，生成"可以当图文读"的笔记——手工课、操作演示、录屏都特别适合：

```bash
# 抽帧（感知去重）+ 已有字幕 → 图文笔记
python3 scripts/extract_frames.py --video demo.mp4 --out-dir runs/demo/frames --mode dedup
python3 scripts/build_notes.py \
  --subtitles runs/demo/subtitles.json \
  --frames runs/demo/frames/frames.json \
  --out-dir runs/demo --describe-frames \
  --model openbmb/minicpm5-2b --vlm-model openbmb/minicpm-v4.6:latest
  # 节点数自动：每 ~45 秒 1 个，钳制 12-60（12 分钟课 16 个节点，41 分钟课 55 个）
  # 也可 --max-frames N 固定
```

每个节点：LLM 小标题 → 帧插图 → VLM 画面描述（`--describe-frames`）→ 旁白浓缩要点 → 原声节选。

**画面描述只写教学信息，不写人。** VLM 提示词明确要求描述乐谱/板书/屏幕文字上的具体内容（音名、和弦、拍号、标注），并**禁止**描述"女士弹琴""手势讲解"这类人物动作；没有可读教学内容时改为点明画面主题。实测同一帧：`女士弹琴，手势讲解` → `乐谱显示音名与和弦`——前者对复习毫无帮助。

> ⚠️ `--describe-frames` 一定要显式给 `--vlm-model` 一个带视觉能力的模型。不给的话它会回落到**文本模型**，看不见画面；现在会打印警告并跳过画面描述，而不是每个关键帧失败一次。

**一份产出两个视图。** 同样的节点、插图、要点，砍不砍原声引用由你选：

| 文件 | 内容 | 适合 |
|---|---|---|
| `notes.md` | 帧 + 要点 + **原声引用** | 核对老师原话 |
| `notes-distilled.md` | 同样的节点和插图，**无原声引用** | 背 / 复习（实测 295 个节点里引用占 53% 字符，砍掉体积少 42%） |

纯享版是从已经算好的结果里再渲染一遍，**不额外调用模型**。`--docx --pdf` 会同时导出两份。`notes.html` 是自包含 base64 单文件（完整版）。

产出：`notes.md`、`notes-distilled.md`、`notes.html`；加 `--docx --pdf` 再出 `notes.docx` / `notes-distilled.docx` / `notes.pdf` / `notes-distilled.pdf`（关键帧嵌入，打印/归档友好；PDF 自动探测系统中文字体，可用 `V2K_PDF_FONT` 指定）。

---

## 🛡️ 稳健性

- **切帧管道流式严格解析**：PGM 帧流按 header 声明的像素数精确读取，绝不扫描像素数据寻找魔数——旧实现在像素字节碰巧含 `P5
` 序列时会**静默截断该视频后续全部关键帧**（长批量下必然偶发，表现为"缺内容"）。回归测试 `tests/test_pgm_stream.py` 用内嵌魔数的真实视频守住此缺陷。
- **尾帧补发（微改动不丢）**：感知哈希阈值只能抓到"足够大"的画面变化——老师在翻页前改一个数字、加一行要点这类**低于阈值的累积微改**会被静默丢弃。检测到翻页时，回看上一相似段的最后一帧：只要它与旧锚点确有差异、自身稳定、且不是新页的近邻，就补发这个"最终状态帧"。三重门限保证手写漂移过程不会帧爆炸（实测：37 分钟课程 28 个尾帧全部命中真翻页前状态，117 个噪声过渡零误发）。回归测试 `tests/test_frames.py`。
### 静默吃内容的 bug（全部已修 + 回归测试）

这一类最危险：文件照样生成、退出码是 0、看不出任何异常，但内容少了一大块。下面每一条都是在 306 个视频 / 61.5 小时的真实批量里才暴露出来的：

| 症状 | 根因 | 修复 |
|---|---|---|
| **标点结尾的字幕段全被丢弃**（一段课程 165 段 → 316 段） | `align_text_ts` 存标点的零宽时间戳时用了**秒**，下游当**毫秒**再除 1000，段首末全部塌到 0 之外 | 单位统一到毫秒 |
| **长课程知识文档在 8000 字被截断**（找回 35% 内容） | `char_limit` 默认 8000 先切片，`long_mode = len(sub) > 13500` 永远不成立——**分块器是死代码** | 自动放开阈值 + map-reduce 分块真正生效 |
| **LLM 不可用被当成成功** | `OLLAMA_MODELS` 指向空目录时 `/api/tags` 返回 200 + 空列表，`/api/generate` 404，脚本 exit 0 | 打印原因 + `is_degraded()` 标记 + **exit 4** |
| **0 段字幕时模型凭空编内容** | 无语音视频照样生成"知识文档"，把「有旋律即兴伴奏」写成「如何使用 Python requests 库发 HTTP 请求」，还伪造 `-[00:03]` 时间戳 | `knowledge_doc_status()` 返回 `ok` / `degraded` / `no-speech`，`no-speech` 不再伪造 |
| **Anki 卡片大面积重复**（1608 张里 145 张重复，最差 117 行 → 11） | `cards_from_qa` 是纯解析器，零去重；而模型很少一字不差重复（"白键之间是全音" vs "白键和相邻白键间隔是一个全音"） | 按**归一化问句**（剥标点/空白/大小写）去重 |
| **图文笔记节点稀疏**（37 分钟课只有 9 个节点） | 节点预算下限卡死：`1-per-90s` 仍钳制在 8-48，而这个库平均片长 12.1 分钟，**下限把大多数视频都按在 8 个节点** | 改成 `1-per-45s`、钳制 12-60。**要动的是下限，不是上限** |
| **`build_notes.py` 把文本模型当 VLM** | `--describe-frames` 不给 `--vlm-model` 就回落成文本模型，每个关键帧失败一次 | `supports_vision()` 前置校验 + 警告并跳过 |

**配套的 ASR 准确度提升**：中文领域同音词用 FunASR Paraformer（44% → 94%），剩下的用 `fix_homophones.py` 兜底——按**带声调拼音**匹配而不是编辑距离，因为 背谱/被谱 连拼音带声调都相同，音频本身就有歧义；而 音阶(yīn jiē) ≠ 音介(yīn jiè)，声调不同就说明音频没歧义。三重护栏保证宁可漏检不可错纠：功能字首尾护栏 + `DEFAULT_BLOCKED = {"实度", "何首"}` + 按偏移量精确替换。实测 20 节课 20 类真错误 46 处，精度 100%。

- **Ollama 死锁规避**：单模型统一配置（视觉+文本同模型）+ 常驻加载，消除多模型切换路径上的服务端死锁。
- 测试：`python -m pytest` — **248 项**，无需网络/模型。

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

| Profile | 触发 | ASR 模型 | VLM（2026-08 阵容） | 文本模型 | 典型机型 |
|---|---|---|---|---|---|
| `tiny` | RAM < 6 GB | tiny | qwen3.5:0.8b (1.0 GB) | minicpm5:Q4_K_M (688 MB) | 树莓派 / 4G 老笔记本 |
| `low` | 6–8 GB 无独显 | base | minicpm-v4.6 (1.6 GB) | minicpm5-2b (1.6 GB) | 上网本 |
| `low-mac` | 6–8 GB Apple Silicon | small | minicpm-v4.6 | minicpm5-2b | M1 MacBook Air |
| `mid` | 8–16 GB | small | minicpm-v4.6 | minicpm5-2b (1.6 GB) | **主流笔记本** |
| `high` | 16–32 GB | medium | qwen3.5:4b (3.4 GB) | 同 VLM ¹ | M2/M3 Pro、16G PC |
| `high-gpu` | NVIDIA ≥ 8 GB 显存 | large-v3 | qwen3.5:9b (6.6 GB) | 同 VLM ¹ | RTX 3060/4060/3090（CUDA+float16 全速）|
| `max` | RAM > 32 GB | large-v3 | qwen3.8:27b (18 GB) | 同 VLM ¹ | 工作站 / 服务器 |

¹ qwen3.5/qwen3.8 是视觉+文本统一模型，这些档位只拉一份权重，第二步直接复用。

模型阵容（2026-08 刷新）：**qwen3.5** 是当代唯一有完整小尺寸阶梯（0.8b/2b/4b/9b，统一视觉+文本，256K 上下文）的 Qwen 代际；**qwen3.8**（原生视频理解）只出 27b+，服务 `max` 档；**面壁 minicpm-v4.6**（1B 端侧效率王牌，CJK OCR 强）守 low/mid 档的视觉位，**minicpm5-2b**（2.5B dense、1.6 GB、OpenBMB 34 项基准均分 53.9，高于 Qwen3.5-4B 的 51.1 而内存约一半）守同档的文本位，**minicpm5**（688 MB）是 `tiny` 档文本模型。旧选型（moondream / qwen2.5vl）已退役为 legacy。

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
<summary><b>Q: 用哪个 python 跑脚本？</b></summary>

venv 固定建在仓库根 `.venv/`。最稳妥的方式是用 `setup_models.sh` 输出的 `run python as:` 绝对路径；或先激活（macOS/Linux `source .venv/bin/activate`，Windows Git Bash `source .venv/Scripts/activate`）再用 `python3`。
</details>

<details>
<summary><b>Q: 跑 <code>build_knowledge.py</code> 卡很久 / 报模型找不到？</b></summary>

第二步会调用**文本模型**（用于摘要/知识点/Q&A），默认由 `hardware_profile.py` 按档位给出：`mid` 及以下为 `openbmb/minicpm5-2b`（约 1.6 GB），`high` 及以上复用 VLM 权重。首次运行时 Ollama 会自动拉取，需要联网和等待；提前手动拉可避免等待意外：`ollama pull openbmb/minicpm5-2b`。想再提质：`--model qwen3.5:9b`。

> ⚠️ 如果 `OLLAMA_MODELS` 环境变量指向了一个空目录，`/api/tags` 会返回 200 + 空列表、`/api/generate` 返回 404——**旧版脚本会照样 exit 0**，你会拿到一份降级产出还以为跑成功了。现在会打印具体原因、给文档打 `degraded` 标记、并以 **exit 4** 退出。
</details>

<details>
<summary><b>Q: 中文视频 ASR 老是"背谱"听成"被谱"、"琶音"听成"爬音"？</b></summary>

三件事按性价比排序：

1. **换 Paraformer**（收益最大）：`python3 scripts/hardware_profile.py --recommend` 看看是不是推荐 FunASR。同一条 30 分钟课实测 44% → 94%。
2. **修错字**：`fix_homophones.py` 按带声调拼音匹配领域词表，把剩下的错字改回来，并同步重写 `.srt`/`.vtt`。先 `--dry-run` 看提案。
3. **热词**：`--hotwords "背谱,视谱,音阶"` 通过 initial_prompt 偏置。

但要知道 2 也有天花板：**背谱/被谱 连拼音带声调都一模一样，音频本身是有歧义的**。所以判断能不能修的标准不是"像不像"，而是"声调一不一样"——音阶(yīn jiē) 和 音介(yīn jiè) 声调不同，说明音频没歧义，放心改。误报几乎都是常用词被切半（曲**是**、声**不**），所以默认带护栏，宁可漏检不可错纠。
</details>

<details>
<summary><b>Q: 图文笔记一个视频只有几个节点 / 画面描述在描述人物？</b></summary>

节点数现在是自动的：每 ~45 秒 1 个，钳制 12–60。想更密或更疏就显式传 `--max-frames N`。

画面描述写人（"女士弹琴，手势讲解"）是提示词的问题，现在的提示词明确要求只写乐谱/板书/屏幕文字上的教学信息，并禁止描述人物动作。如果画面描述**整段消失**，多半是 `--describe-frames` 没配 `--vlm-model`，回落到了看不见图的文本模型——现在会打印警告并跳过。
</details>

<details>
<summary><b>Q: Anki 牌组里一堆几乎一样的卡片？</b></summary>

现在 `cards.csv` 会按**归一化问句**去重（剥标点/空白/大小写），因为模型很少一字不差重复——"两个白键之间是全音" 和 "白键和相邻白键间隔是一个全音" 是同一张卡。旧牌组不会自动修，需要离线跑一次去重重建。
</details>

<details>
<summary><b>Q: 跑 <code>ollama not found</code> / <code>ffmpeg not found</code>？</b></summary>

回【第 1 步】把对应工具装上并确认在 PATH 里：`ollama --version && ffmpeg -version`。Ollama 装好后若未常驻，`setup_models.sh` 会自动 `ollama serve` 拉起；若仍失败，手动开一个终端跑 `ollama serve`。
</details>

<details>
<summary><b>Q: 模型 / pip 下载很慢或超时？</b></summary>

国内网络建议挂代理：`export HTTPS_PROXY=http://127.0.0.1:7890`（端口换成你自己的）。pip 可换镜像：`pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`。
</details>

<details>
<summary><b>Q: Windows 上能跑吗？</b></summary>

可以，**Git Bash 原生支持**（v1.1 起全面适配：自动映射 `USERPROFILE`、识别 `Scripts/` venv 布局、拦截 MSYS/mingw python 陷阱），也可用 WSL。Ollama 用官方安装包，ffmpeg/python 用 `winget` 安装。
</details>

---

## 🔒 隐私与留痕

- **默认全程本地**：视频文件、抽帧、字幕、知识产物始终留在你机器上的 `runs/<时间戳>-<视频名>/`，**默认路径不联网、不调用任何云 API**。FunASR Paraformer、faster-whisper、Ollama 全部跑在本机。
- **云端后端是显式 opt-in，且会把音频/文本发出去**：如果你显式传了 `--backend openai-api` / `mimo-asr`（ASR），或 `--api-base`（知识文档，兼容 OpenAI 的 `/chat/completions`），**对应的音频或字幕文本会上传到那个第三方端点**。这不是默认行为，也不会自动发生——不传这些参数就完全本地。密钥从环境变量读（`--api-key-env`），不会出现在命令行历史里。
- **仓库只跟代码**：本仓库是 skill 本身（脚本/文档/模板）的版本管理，**不包含任何视频或处理产出**——`runs/` 已在 `.gitignore` 中忽略。代码与配置的修改都有 git 历史可追溯。
- **本地复现**：要复现某次结果，在本地 `runs/<...>/` 里查看当次用的参数和产出即可（按需自行写 `manifest.json` 记录，但默认不入库）。

`example/` 目录提供一份用 ffmpeg 合成视频跑通的示例产出（无真实数据，仅供演示结构与字段）。

---

## 📂 项目结构

```
video2knowledge/
├── SKILL.md                       # 主控文档（流程编排 + 留痕规范）
├── scripts/
│   ├── hardware_profile.py        # 机型检测 → 配置档（单一真相源，跨平台探测链）
│   ├── setup_models.sh            # 幂等：检测机型 + 拉模型 + 建 venv（macOS/Linux/Win Git Bash）
│   ├── asr_caption.py             # 路径 2：faster-whisper → 字幕（--hotwords 术语偏置）
│   ├── mm_caption.py              # 路径 1：VLM 逐帧 → 字幕（OCR 文本变化门控）
│   ├── extract_frames.py          # ffmpeg 抽帧（interval/dedup：稳定帧选取+空白门控+dual 哈希+簇式预算）→ frames.json
│   ├── merge_visual.py            # 路径 3：ASR × VLM 按时间戳融合 → merged.json（re-attach 回退 + 语义对齐校正/弱归属标记）
│   ├── hotwords_from_ocr.py       # 路径 3 反馈：OCR 术语 → 热词表 + ASR 覆盖率检查（可触发重转写）
│   ├── build_knowledge.py         # 第二步：字幕 → 知识文档/HTML/CSV（--format docx/pdf 可导 office）
│   ├── build_notes.py             # 2.4：图文笔记（关键帧 × 旁白）→ notes.md/.html（--docx/--pdf）
│   ├── md_export.py               # Markdown → DOCX/PDF 导出器（python-docx + fpdf2，CJK 字体自检）
│   ├── gen_apkg.py                # 2.3：CSV → Anki .apkg
│   └── batch_run.py               # 批量跑整个课程库（两级流水线、断点续跑、优先级排序）
├── references/                    # 详细文档（按需加载）
│   ├── hardware-profiles.md
│   ├── path1-multimodal.md
│   ├── path2-asr.md
│   ├── path3-fusion.md
│   ├── templates.md
│   └── outputs.md
├── assets/default-template.md     # 内置默认知识文档模板
└── example/                       # 合成视频的示例产出（无真实数据）
```

> 处理真实视频时，产出会写到本地 `runs/`（已 gitignore，不入库）。

---

## 🧠 设计取舍

- **全本地推理**：用 Ollama 跑 VLM/文本模型、faster-whisper 跑 ASR，视频内容不离开本机，隐私可控、留痕可复现。
- **小模型优先，按档位自适应**：2026-08 阵容里 0.8b–4b 级模型覆盖 6–32GB 机器，8GB 机器跑得动；文本模型同样走档位表——`mid` 及以下用 minicpm5-2b（1.6 GB），`high` 及以上复用 VLM 权重，不额外下载。Q&A 在过小模型上偶有偏差时，换 `--model qwen3.5:9b` 即可显著改善。
- **CLI 优先，可被 skill 调用**：所有脚本带 `--help`、不硬编码路径、幂等。

---

## 📜 License

[MIT](LICENSE) © CacinieP
