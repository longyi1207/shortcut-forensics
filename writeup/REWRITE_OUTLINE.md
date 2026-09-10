# 9月10日重写提纲（讨论稿）

这是给我们一起重写用的骨架，不是正文。数字位置用「□」占位，早上 `scripts/morning_report.sh`
生成的 `RESULTS_2026-09-10.md` 填进去。每一节后面标了它靠哪张表/图支撑，以及目前证据够不够。

## 标题（候选）

- Prompting and contrast-direction steering do not move the same thing: a per-factor test on an agentic shortcut task
- 备选更短：What a "don't be like this" line does that the concept's direction does not

## 1. 问题（半页）

一个 agent 在 258 个 mypy 错误 + pre-commit hook 的任务里会走捷径（伪绿）。我们有五个候选「动机」因子
（tedium, desperate, shortcut, completion_drive, disapproval），每个因子都有：(a) 一条 contrast-pair 拟合出来的
方向 d，(b) 一条自然语言指令「你不要/要这样」。问三件事，逐因子回答：

1. 指令放进 user prompt，捷径率变多少？（vLLM，同夜 baseline）
2. 用 d 做 steering（ablate 或 +add，按符号），捷径率变多少？（HF，HF baseline）
3. 指令在激活上留下的足迹，和 d 的重叠有多少？（prompt 末尾位置 + decision token 位置）

为什么值得问：steering 文献常默认「找到了概念方向 = 能像 prompt 一样控制行为」；这里五个因子各自检验。

## 2. 设置（半页，全是已有事实）

- 环境、模型（Qwen3.5-9B，3:1 GatedDeltaNet/attention）、judge（Azure，is_shortcut / workaround_type / capability_ok）。
- 方向：mean-diff，最后 token，L19（disapproval 用 L17 refit）；验证门槛（held-out acc、lexical scramble、plant）。
  **必须写明**：tedium L19 没过 plant gate（L8/L12 过了）；旧稿写成 pass 是错的。
- 指令：五条同长度同格式（IMPORTANT: …），外加一条同长度无概念内容的 neutral 控制线。
- 同夜同后端原则：vLLM 只和 vLLM 比（id 前缀 rvpd/rvpt），HF 只和 HF 比。原因：跨后端 baseline 差异（13-20% vs 34%）。

## 3. 结果一：prompt 的行为效应（表 + 图 prompt_vs_steer.png 左半）

| 因子 | prompt 捷径率 | vs baseline □% | p |
|---|---|---|---|
| tedium | □ | | |
| desperate | □ | | |
| shortcut | □ | | |
| completion_drive | □ | | |
| disapproval | □ | | |
| neutral 线 | □ | | |

要点句（待数据）：哪些指令真的降低捷径率、neutral 线是否也动（如果 neutral 也动，效应里有「任何长指令」的成分）。

## 4. 结果二：direction steering 的行为效应（同一张图右半）

已有：ablate_tedium 10% (n=60)，ablate_desperate 17% (30)，ablate_shortcut 25% (32)，add_pos_disapproval_L17 21% (34)，
ablate_completion_drive □ (目标 30)，HF identity 20% (20)。控制：ablate_random 30% (10)，随机方向 +add（pc_add_rand19，目标 16）。
其它层（tedium L8/L12、shortcut L23-L29、disapproval L8/L12）一并列出，说明不是挑层。

要点句：steering 效应小且 CI 宽；和随机方向控制不可区分（如果数据如此）。诚实写：HF baseline 只有 n=20。

## 5. 结果三：激活重叠（图 geometry.png）

- prompt 末尾：概念自己的 plus-pole plant 能沿 d 推动（desperate +2.4、shortcut +1.9、tedium +0.85，null ≈ 0±0.4），
  指令几乎不动（−0.25 到 +0.38）。completion_drive 和 disapproval 的 d 连自己的 plant 都推不动（任何层）。
- decision token（指令在 10-40 轮之前）：所有文本的 cos 都在 ±0.04 内，d 在这里几乎不可读，所以「正交」单独不说明问题。
- 指令之间的足迹彼此相似（cos 0.35-0.74）：五条指令共享一个「被叮嘱」的方向，而不是各自的概念方向。

要点句：指令走的不是 d 这条坐标；它在行为上有效（结果一）但在几何上和 d 无关（结果三）；d 在几何上可被概念文本激活，
但拿它做 steering 行为效应小（结果二）。三件事拼起来就是标题。

## 6. 不成立的东西（明确写，半页）

- decision-point readout s 不跟 judge 标签（AUROC 0.56-0.59，方向还反了）：旧稿第 5-7 节的句子级/电路级结论撤回。
- tug-of-war 不是主线，只作为附录或删掉。
- tedium L19 plant gate 失败；19 vs 258 错误数、8×H100→A100、字数等旧稿错误清单。

## 7. 相关工作（补）

arXiv 2604.09839、2605.03907、2605.10664（prompt 与 steering 的关系、agentic reward hacking 的方向）。

## 8. 局限与下一步（短）

n（每格 60-90 / 30），单模型，单任务，judge 单模型；下一步：把指令足迹本身作为方向去 steer（prompt_delta 已有初步）。

## 字数与人声

Exec summary ≤ 619 词（旧稿 1035 超了）。Form 答案由 LY 亲自写。
