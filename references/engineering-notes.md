# Implementation notes and measured examples

Read only when explaining design choices or historical regressions. These are
sample-specific observations, not guarantees for a new video or machine.

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
- 测试（无需网络/模型，抽帧回归需要系统 `ffmpeg`）：
  ```bash
  python -m pip install -r requirements-test.txt
  python -m pytest -q
  python tests/test_pgm_stream.py
  ```
  GitHub Actions 在 Python 3.11 / 3.12 上执行上述检查。默认运行依赖统一列在
  `requirements.txt`；`setup_models.sh` 会安装并补齐，包括错词修复的 `pypinyin`。
  Linux 中文 PDF 导出还需字体，例如 `sudo apt-get install fonts-wqy-microhei`。

维护者发布：更新 `.github/release.json` 和对应发布说明，在合并提交信息中加入
`[release]`。仅主分支 push 且两组回归通过后，CI 才发布指向该提交的 GitHub Release；
普通推送和 PR 不发布。

