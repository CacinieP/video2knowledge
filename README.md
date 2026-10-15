# 🎬 video2knowledge

把课程、教程和录屏变成可以核对来源的笔记。安装成 **Agent Skill** 后，向 agent
提供视频路径和目标，它会选择语音转写、画面理解或音画融合，执行并检查产物。
默认使用本地模型；云端音频、文本和画面处理均需选择相应端点。

```text
用 video2knowledge 把 /Users/me/Movies/lecture.mp4 做成中文图文笔记，保持本地。
这个录屏没有声音：/Users/me/Movies/demo.mp4。整理画面步骤，输出 HTML。
继续处理上次中断的中文课程库，沿用 /Users/me/runs/course，告诉我失败了哪些。
```

默认笔记交付 Markdown + HTML；字幕、Anki、DOCX/PDF 按任务选择。
遇到缺模型、内容不足或部分失败，agent 应交代缺失项和恢复方式，不能把空壳文件当完成。

## 先看产物

仓库保留了 NASA 公有领域视频的历史示例（示例产物，不代表新版本完整实测）：
[知识笔记](example/knowledge.md) · [HTML 源文件](example/knowledge.html) ·
[卡片 CSV](example/cards.csv) · [Anki 牌组](example/cards.apkg)。
下载 HTML 后可离线打开；Anki 需桌面端导入。示例中的一条卡片：

| 问题 | 答案 |
|---|---|
| How does the speaker feel about leading a team? | Expresses humility and questions his own worthiness. |

有声幻灯片可同时保留讲解与画面信息；无声录屏可生成带关键帧的纯画面笔记。
结果依赖视频质量、采样和所选模型，请核对重要数字、公式及卡片答案。
英文标题／正文可用 `--lang en`；当前本地画面图注仍是中文，英文专用笔记默认省略图注、保留原始关键帧。

## 安装为 Skill

需要 Python 3.11+（或 uv）、ffmpeg/ffprobe；本地推理还需要 Ollama。
先将稳定版安装到你的 agent 实际使用的 skills 目录，目录名保留 `video2knowledge`。
例如 Claude Code：

```bash
git clone --branch v1.9.1 https://github.com/CacinieP/video2knowledge.git ~/.claude/skills/video2knowledge
```

其他客户端使用自己的技能发现目录，例如 ZCode 的 `~/.zcode/skills/`、Cursor 的
`~/.cursor/skills/`；以客户端配置为准。仓库自带 `SKILL.md` 和 `agents/openai.yaml`，
不要求把全部文档塞进系统提示词。如果客户端未发现技能，先核对安装目录/加载设置。

然后对 agent 说：

> 使用 video2knowledge 检查本机依赖，为本地处理准备环境，再把这个视频做成笔记。

首次准备会创建虚拟环境并下载模型，耗时取决于机器和网络，**不保证五分钟完成**。
如果不允许下载或启动服务，请在请求里说明；agent 应先报告缺项。
手动准备时，切换到安装目录：

```bash
cd "/absolute/path/to/video2knowledge"
bash scripts/setup_models.sh
```

中文 Paraformer 后端另加 `--with-funasr`，会使用独立虚拟环境；云端 ASR SDK 另加
`--with-openai-client`，都不是默认安装。云端专用/禁止下载的任务不要盲跑本地模型安装。
系统安装和解释器定位详见 [运行环境](references/runtime.md)。

## Agent 应怎样选择路线

| 视频与任务 | 路线 | 默认交付 |
|---|---|---|
| 有讲解，且需要屏幕、板书、表格或演示内容 | 音画融合 | 知识笔记 + 图文笔记 |
| 有清晰讲话，只需要讲了什么 | ASR | 字幕 + 知识笔记 |
| 无可用讲话，或明确只看画面 | 纯画面 / OCR | 关键帧图文笔记；按需导出画面文字 |

音乐音轨不等于讲话；画面描述不等于老师原话。批处理入口目前固定中文音画融合，
英文库和纯无声库需要逐视频走对应脚本，不能照抄中文批处理命令。

给 agent 的完整操作约定见 [SKILL.md](SKILL.md)。它应在结束时报告：

- **完成／部分完成／阻塞**，以及产物路径或链接；
- 采用的路线、实际模型和本地／云端处理情况；
- 未生成的格式、降级、已知遗漏，失败时的续跑命令。

## 手动使用：可靠的路径约定

不要依赖 agent 的不同终端调用共享 `source .venv/bin/activate`。
每次调用使用解释器和脚本的绝对路径。以下是一个 Bash 调用块：

```bash
SKILL_ROOT="/absolute/path/to/video2knowledge"
PY="$SKILL_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$SKILL_ROOT/.venv/Scripts/python.exe"
"$PY" "$SKILL_ROOT/scripts/hardware_profile.py" --json
```

自定义 venv 时使用安装脚本打印的 `run python as:` 路径；Windows 采用 Git Bash 和
原生 Python。下面按需读取一条配方，不必读完全部参数：

| 要做什么 | 操作说明 |
|---|---|
| 有声课程／采访的文字笔记 | [ASR 配方](references/path2-asr.md) |
| 无声录屏的图文笔记／OCR | [纯画面配方](references/path1-multimodal.md) |
| 讲解与屏幕内容结合 | [音画融合配方](references/path3-fusion.md) |
| 中文课程库、失败重跑 | [批处理与续跑](references/batch-resume.md) |
| Anki、DOCX/PDF、输出验收 | [产物说明](references/outputs.md) |
| 可选云端处理 | [端点与数据边界](references/cloud.md) |
| 模型与资源选择 | [硬件参考](references/hardware-profiles.md) |
| 自定义笔记结构 | [模板说明](references/templates.md) |

默认笔记配方使用 `--format html`，同时生成 Markdown 与 HTML。
`build_knowledge.py --format all` 不会替你生成图文笔记或 Anki 包，二者有独立步骤。
PDF 是可选项：安装 `fpdf2` 后还需中文字体，例如 Linux 的 `fonts-wqy-microhei`。
HTML 为轻量渲染；当前知识 HTML 的时间戳不提供可用的跳转目标或视频播放集成。

## 重跑与升级

同一次任务续跑应复用原输出目录和配置。批处理校验 `run_manifest.json` 的哈希，
保留可复用 ASR 检查点；配置变化会清理旧知识分块缓存，配置未变的失败重试仍可续块。
`run_manifest.json` 不记录可读参数；需要审计时另外写本地 `manifest.json`。

**从 v1.8.x 及更早升级到 v1.9.x：**旧无哈希输出目录会保留，但不自动复用，
首次运行会重新处理到新目录。Anki 身份规则也变了，旧牌组需手动迁移，首次新版导入
可能出现重复。对同一来源使用稳定 `--source-id`，不要为不同视频复用同一个 ID。

源视频校验使用路径、大小和纳秒修改时间，不扫描整个视频；同名模型权重被替换也
不会自动识别。完成标记不等于事实准确或所有可选格式齐全，仍须核对请求的产物。

## 本地与云端

默认配方在本机处理；选择云端 ASR 会发送音频，云端知识/笔记会发送字幕或融合文本，
云端 VLM 画面描述还会发送所选关键帧图片。远程 Ollama 同样会接收对应文本和图片。
已有环境变量也可能启用端点，因此不能只检查是否显式传入 `--api-base`。
详见 [云端说明](references/cloud.md)。

密钥从配置的环境变量或凭据存储读取，不放进命令行或产物清单。
`runs/` 中的视频、字幕和产物默认被 git 忽略；仓库维护不应提交用户媒体。

## 验证与维护

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
python tests/test_pgm_stream.py
```

抽帧回归需要 ffmpeg；中文 PDF 测试还需支持的字体。CI 在 Python 3.11 / 3.12 上运行。
这些检查覆盖代码行为，**不等于真实视频内容准确率，也不等于跨客户端安装实测**。
文档示例还需检查 CLI 参数及路线边界。历史实测和修复原理移到
[工程笔记](references/engineering-notes.md)，其样本数字不作为新任务的质量保证。

维护者发布：更新 `.github/release.json` 和对应说明，合并信息加入 `[release]`；
主分支两版本 CI 通过后，才发布指向该提交的 GitHub Release。

MIT License，见 [LICENSE](LICENSE)。
