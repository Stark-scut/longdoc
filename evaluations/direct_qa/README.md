# Qwen3.5-9B：MMLongBench-Doc 全文页图直接评测

## 仅证据页 + OCR + crop 的 e2e 对照实验

新入口 `run_gold_evidence_ocr_crops.py` 对应原入口
`run_answerable.py --gold-evidence-only`，区别仅在输入：

- 原模式：GT 证据整页图。
- 新模式：同样的 GT 证据整页图，以及**仅这些页**的 OCR blocks 和可用 crop。

新模式直接复用 `agent.documents.Page.parts()`，与 Reasoner 的证据内容渲染一致。
按页发送整页图，再发送该页各 block 的 OCR 和已有 crop，不补入其他页面。
不存在 crop 的 block 仍发送 OCR；选定页缺少 OCR 记录时直接报错。
它运行一次 e2e 问答，不运行 Locator/Reasoner 循环，不使用 Reasoner 的提示词或 JSON 输出协议。
系统使用的问答指令与原 e2e **完全相同**：

```text
Read all pages of the document and answer the question based on the document. If the question cannot be answered from the document, answer 'Not answerable'.
```

问题放在证据内容之后，仍为 `Question: ...`。短答案抽取提示词、官方评分函数也保持不变；
回答和抽取默认均为 temperature=0、seed=0、关闭 thinking。
GT 答案不进入模型输入，GT 页号只用于离线选择证据页。这是使用标注证据的对照实验。

在本服务器，从仓库根目录先检查全部数据：

```bash
python evaluations/direct_qa/run_gold_evidence_ocr_crops.py \
  --agent-dir output/agent/3090_8gpu_tp2_memory_c4_mm3 \
  --gpu-ids 0,1,2,3 \
  --concurrency 4 \
  --output-dir output/mmlongbench/qwen35_gold_ocr_crops \
  --dry-run
```

去掉 `--dry-run` 启动评测。小规模验证可使用 `--gpu-ids 0 --limit 4` 和独立输出目录；
切回多 GPU 全量评测时使用新目录，避免混用不同分片结果。
这里每张卡部署一个完整 BF16 / TP=1 模型，沿用原 e2e 多副本入口，适用于空闲 A100/A6000；
`--agent-dir` 只提供冻结题集、页面清单和数据/模型路径，不沿用该 Agent 实验的 TP=2 部署。
迁移服务器时可覆盖 `--model-path`、`--images-dir`、`--blocks`、`--crop-root`。

入口沿用原证据页评测的筛选规则：只评可回答题，GT 页号从 1 起始转换为 0 起始；
空证据或越界题在启动时列出并排除。以以上冻结实验为例，859 道可回答题中保留 838 道。
新模式禁止截掉页面/crop；整页保持原分辨率，crop 沿用每图 1048576 像素上限，
超大 crop 与 Reasoner 一样由服务端预处理缩放。
整页和 crop 都计入图片数量上限和视觉 token 估算，
文本和模板也会占用上下文，最终以服务端检查为准，超限记录失败。
相同 PDF 固定分配给同一 GPU，沿用滚动 PDF 调度及 Prefix/MM cache；不同证据页集合不保证完整前缀命中。

结果沿用 `merged.json`、`merged.summary.json`、`merged.txt` 和 `comparison.txt`。
逐题新增 `generation_input`，记录实际证据页号、OCR block 数、crop 数和源图片路径；
不额外复制图片。配置记录输入模式、OCR SHA256 和图片清单指纹（路径/大小/修改时间），
阻止混用图片模式或数据变更前的回答。
已有输出请使用新目录。原 `--gold-evidence-only` 行为保持不变。

如使用已有服务而不自动部署，可用 `evaluate_qwen35_mmlongbench.py`
搭配 `--evidence-ocr-crops --blocks ... --crop-root ...`；其输入 JSON 每题必须显式包含
`input_page_indices: [0, ...]`，同时不要启用 `--truncate-long-documents`。

## 当前 A100 环境的 Prefix cache 修复与文档分片

本地 vLLM 0.24.0 的多图 Mamba 对齐停滞已应用兼容补丁，原分片配置已在 A100
完成前 256 题、零错误。排错步骤、可回退补丁和结果证据见
[PREFIX_CACHE_FIX.md](PREFIX_CACHE_FIX.md)。

新的测试配置将图片预处理缓存设为 4 GiB，并将相同 PDF 的问题固定到同一 GPU，
每份 PDF 先完成首题再并发其余题目，避免各卡重复计算同一文档的冷前缀：

```bash
bash /media/data5/leizh/longdoc/longdoc/evaluations/direct_qa/test_prefix_fix.sh --limit 256 --timeout 300
```

对应新参数为 `--shard-strategy document --document-warmup --mm-processor-cache-gb 4`。
GPU、输入、模型、页面设置均由脚本指定；结果写入独立的 `qwen35_a100_doc_prefix_mm4`。
新配置的实际提速和图片缓存稳定性仍需 GPU 验证，按文档分片不能复用旧按题分片目录。
下面保留原双卡入口及背景说明。

推荐入口是 `run_replicas.py`：任意数量 GPU，每卡一个完整 BF16 / TP=1 模型，默认无 CPU 权重 offload。
这里运行直接问答，不运行 Agent：文档页图在前、原始问题在后，生成回答、抽取短答案，再与 GT 计算官方 ACC / F1。
默认全文页图模式的模型请求不包含 GT 答案、gold evidence 标记或 OCR。

## 四张 A6000，每张卡并发 8

在已配置好 vLLM / PyTorch / Transformers 的环境中，从仓库根目录运行。
客户端额外依赖可用 `python -m pip install -r evaluations/direct_qa/requirements.txt` 安装。
模型、`samples.json` 和 `images_1024` 需在另一台服务器单独准备；它们不在 Git 仓库内。
如果数据仍在仓库原相对位置，可省略 `--input` 和 `--images-dir`。

```bash
python evaluations/direct_qa/run_replicas.py \
  --gpu-ids 0,1,2,3 \
  --model-path /path/to/Qwen3.5-9B \
  --input /path/to/mmlongbench/samples.json \
  --images-dir /path/to/mmlongbench/images_1024 \
  --max-model-len 262144 \
  --truncate-long-documents --input-token-limit 260000 \
  --concurrency 8 --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --enable-prefix-caching --mm-processor-cache-gb 1 \
  --cpu-offload-gb 0 \
  --output-dir output/mmlongbench/qwen35_a6000_4gpu
```

- 四卡端口默认 8047、8048、8049、8050。支持非连续卡号，例如 `--gpu-ids 1,4,5`；端口可用 `--ports` 逐一指定。
- `--concurrency` 是**每卡**客户端并发，四卡设置 8 时最多同时处理 32 题；`--max-num-seqs` 是每卡 vLLM 调度上限。
- 使用原生上下文 262144，不自动启用 YaRN；不指定 `--max-model-len` 时读取本地模型配置。
- 回答与短答案抽取均关闭 thinking。默认输出预算分别为 8192 / 2048 token。
- 默认完整 BF16 权重驻留每张 GPU，不量化。KV 容量和实际并发仍受显存限制，8 路满长上下文不保证能同时运行。
- Prefix caching 默认开启，并为 Qwen3.5 设置 `--mamba-cache-mode align`。MM processor cache 默认 1 GiB/进程，在 CPU 内存中，与 GPU KV cache 不同。
- `images_1024` 保持源分辨率，仅做模型需要的 32 像素对齐，不自动缩图。默认 eager；`--no-enforce-eager` 可另测 CUDA Graphs。
- 短答案默认由同一副本提取；可显式设置 `--extractor-base-url` / `--extractor-model`。抽取器变化需要新目录。
- 新服务器不需要复制旧评测源码：评分函数与抽取提示词优先读取归档，归档不存在时读取仓库自带的 `agent/metrics/mmlongbench.py` 和 `agent/prompts/extract.txt`。这些文件不启动 Agent。

## 超长文档：只保留前部整页

新入口默认开启 `--truncate-long-documents`，输入目标上限为 260000 token。
上下文包含输入和输出，因此默认 8192 输出预算下，有效输入预算是 `min(260000, 262144 - 8192) = 253952`。
进一步为问题、模板、页号和图像标记预留空间，再按视觉 token 估算保留可容纳的连续前部页面。
正常文档仍输入全部页面；超长文档从末尾整页丢弃，不切碎图片、不缩图。截断后证据可能缺失，此结果属于前缀截断评测，不能标为全部页面评测。

初始预算按图像对齐尺寸估算；后端仍报上下文超限时逐次删除最后一页重试，最多 8 次。
若成功请求返回的 `usage.prompt_tokens` 超过有效输入预算，也会丢弃该次输出、记录其用量后缩短输入重试。
记录中的 `actual_prompt_tokens` 来自服务端；服务未报告用量时只保留估算值。

逐题 `input_truncation` 记录策略、原页数、保留页数、丢弃页数、保留的 0 起始页号、估算/实际 token 和重试信息。
`merged.summary.json` 的 `truncated_inputs` 统计截断题数。
需要严格保留所有页、超限记失败的旧行为时，使用 `--no-truncate-long-documents`。
改变截断开关、输入预算或截断模式下的上下文长度，必须使用新输出目录，避免混用不同输入的答案。

## 进度、日志与中断续跑

上述命令加 `--dry-run` 可先检查数据及完整启动计划，不启动服务、不写结果。
加 `--limit 4` 可先测试四题；随后移除 limit、沿用同一目录完成全量。
主进度条显示成功/失败、每卡已处理数量、题/小时与 ETA；ETA 会随文档长度波动。
相同命令可中断续跑：完整题目跳过，已生成回答但抽取失败的题目只重试抽取；Ctrl+C 清理本入口启动的服务进程组。
保持副本数和端口一致；改变副本数需新目录。并发和缓存容量可在停止入口后调整。

输出统一放在 `--output-dir`：

- `shardN.json`：逐题原回答、抽取答案、官方分数、用量、状态及截断记录。
- `serverN/server.log`、`serverN/command.txt`、`serverN.launcher.log`：每副本服务日志和实际命令。
- `evalN.log`：各评测进程日志。
- `run_config.json`、`run_history.jsonl`：配置与续跑历史；`progress.json`：进度快照。
- `merged.json`、`merged.summary.json`、`merged.txt`：合并答案、ACC/F1 与分层报告；失败和未完成计零。

```bash
# 另一个终端只读查看进度
python evaluations/direct_qa/run_replicas.py --watch \
  --output-dir output/mmlongbench/qwen35_a6000_4gpu

# 停止任务后，重新合并已保存的结果
python evaluations/direct_qa/run_replicas.py --merge-only \
  --output-dir output/mmlongbench/qwen35_a6000_4gpu
```

以下保留旧入口说明。`run_qwen35_mmlongbench_2gpu.py` 是兼容入口，默认仍使用旧的 131072 上下文、每卡并发 1、无截断、Prefix/MM cache 关闭；新实验请使用上述入口。

## 使用现有 1024 页图分辨率、两卡各一个模型

新入口 `run_qwen35_mmlongbench_2gpu.py` 自动启动两个独立的 vLLM 服务，
每张卡加载一份 BF16 Qwen3.5-9B（TP=1）。按原始题号取模分为两份，各 541 题，
端口默认为 8047、8048；默认 GPU 1、2，也可以显式指定另外两张空闲卡。
默认关闭 thinking，每个副本同时处理 1 题，短答案抽取默认仍使用同一副本。
**现在默认 CPU offload=0，每张 GPU 都保存一份完整模型权重。**

主终端现在显示全局进度条：成功/失败题数、每卡已处理数、题/小时、已用时间和 ETA。
每 5 秒刷新，`--progress-interval` 可修改。已处理数包含成功和失败；
ETA 按两个分片的实际吞吐分别估算，取较慢者的剩余时间。
长短文档的处理时间差异很大，初期 ETA 只作参考；重启后的 ETA 会重新估算。

```bash
conda activate qwen35
cd /media/data1/leizh/longdoc

# 只检查数据、分辨率、分片和命令，不启动服务。
python evaluations/direct_qa/run_qwen35_mmlongbench_2gpu.py --gpu-ids 1,2 --dry-run

# 先测试两题，每张卡一题。服务随脚本启动，完成后自动关闭。
python evaluations/direct_qa/run_qwen35_mmlongbench_2gpu.py --gpu-ids 1,2 --limit 2

# 同一目录继续全量评测，复用冒烟测试已完成的答案。
python evaluations/direct_qa/run_qwen35_mmlongbench_2gpu.py --gpu-ids 1,2
```

不自动终止其他服务。原有两卡 TP=2 服务如果仍占着所选 GPU，需要先在其终端结束，
或使用 `--gpu-ids 4,6` 指定两张空闲卡。完整任务期间服务和评测由这个入口统一管理。
中断后可重跑同一命令续评；不要同时启动两个使用相同输出目录的入口。

如果旧版本任务已经在跑，修改脚本不会改变该进程。在另一个终端运行只读监控，
即可显示已有任务的总体进度和 ETA，无需重新加载模型：

```bash
python evaluations/direct_qa/run_qwen35_mmlongbench_2gpu.py --watch
# 如原任务使用了自定义 output-dir 或 limit，监控时沿用。
```

Ctrl-C 退出监控不会停止评测；watch 不写结果、不启动或终止服务。
兼容旧任务：从已有分片 JSON 和 eval 日志里的进度条计算进度。

默认页图目录仍是 `images_1024`，**保留这些文件的现有分辨率**，不是更大的 `images` 渲染图。
客户端和服务端均使用 `min_pixels=1024, max_pixels=1048576`，
启动前读取所有源图尺寸，确认对齐后的面积不超过上限；上限不足就报错，避免悄悄缩图。
视觉处理仍需要按 Qwen3.5 的 `patch_size=16, merge_size=2` 对齐到 32 的倍数，
因此不能保证逐像素不发生重采样。每道题仍发送全部页面。

24GB 单卡不能把“装下权重”视作“所有文档都能推理”：本地权重文件总大小约 18.0 GiB，
还需要视觉计算、激活、KV cache、线性注意力状态和 CUDA 工作空间。
旧版入口默认每卡 `--cpu-offload-gb 6`，是为了给长上下文留下 KV 空间，
但 CPU 权重访问引入了明显的 PCIe 开销。现在默认改为 **0**，完整 BF16 权重留在各自 GPU，
不做量化；不会因显存不足自动开启 CPU offload。
只有用户显式传入正数 `--cpu-offload-gb` 才会启用 offload。
模型加载时仍然需要使用 CPU 内存读取文件，这与推理时 offload 权重不同。
24GB 是否能同时容纳完整权重、视觉计算和指定 KV 容量仍需以启动日志为准。

默认 `max_model_len=131072`（旧版 196608）、`max_num_seqs=1`、`max_num_batched_tokens=1024`、
GPU memory utilization=0.95、eager。`--concurrency` 是每卡客户端并发，
`--max-num-seqs` 是每卡服务端调度上限；两者需要一起调整才能测试推理批处理。
例如只把 concurrency 提高到 4、max-num-seqs 仍为 1，主要只会增加服务端排队。

GPU 利用率 100% 本身不能证明计算或内存带宽已被充分利用。
完整权重放回 GPU 后，留给 KV 的空间减少；先试每卡 2 并发，看题/小时是否提高，
同时检查 server 日志是否出现频繁 preemption/recompute。
提高并发不保证加速；16 并发叠加长文档会增加 KV 压力和单题延迟。

并发、max-num-seqs、max-num-batched-tokens、CPU offload 和上下文容量
可在停止旧入口后调整，并沿用同一输出目录续跑，已完成题目不会重新生成。
每次启动的配置写入 run_history.jsonl；子分片元数据保留上下文变更历史，
逐题 generation.context_limit 记录生成时的上下文上限。
改变容量后仅检查尚未生成回答的请求，不截断或修改已经保存的回答。
模型、图像分辨率、提示词、输出预算、thinking 或抽取器变化仍需新结果目录。

```bash
# 先 Ctrl-C 停止原入口并等待退出，再用同一目录续跑：默认 offload=0，每卡并发 2。
python evaluations/direct_qa/run_qwen35_mmlongbench_2gpu.py --gpu-ids 1,2 --max-num-seqs 2 --concurrency 2

# 若要从零评测，使用独立结果目录；先每卡测试一题。
python evaluations/direct_qa/run_qwen35_mmlongbench_2gpu.py --gpu-ids 1,2 --limit 2 \
  --output-dir output/mmlongbench/qwen35_1024_gpu_only_fresh
```

数据中最长的 468 页文档仅图片就约 374400 个视觉 tokens，
超过上述上下文，也超过模型原生 262144 上下文。
在新默认 131072 上下文、8192 输出预算下，预检查共 **7 份文档、48 道题**超限，
若无已保存回答，这些题会明确记失败并计零，不删除页面、不降低分辨率。
旧版 196608 上下文时只有最长文档的 5 道题明确超限。
因此纯 GPU 模式不能当作可处理全部文档的同条件完整复现。
若沿用旧结果，较大上下文下已经生成的完整回答可以保留，不重跑。
提高上下文可尝试覆盖更多题，但单卡可能因 KV 容量不足而启动失败；
不会自动缩图、启用 offload 或静默降低上下文。
保留超限失败时 `evaluation_complete` 为 false，程序退出码为 1；
需查看错误类型区分预期超限与额外的启动、OOM 或请求失败。

默认输出目录 `output/mmlongbench/qwen35_1024_2replicas_bf16/`：

- `shard0.json` / `shard1.json`：每卡逐题保存的回答、抽取、评分和状态，保留原始题号。
- `eval0.log` / `eval1.log`：每卡评测日志。
- `server0/` / `server1/`：实际服务命令和日志；`.launcher.log` 包含启动前检查。
- `merged.json` / `merged.summary.json` / `merged.txt`：两个分片合并后的全局结果与评分。
- `run_config.json`：部署与评测配置，不包含 API 密钥。
- `run_history.jsonl`：每次启动的配置与时间，记录并发调整历史。
- `progress.json`：新入口运行期间刷新的全局计数、每卡吞吐和 ETA。

程序结束或中断时合并结果；运行期间以主终端进度条和分片文件为准。
也可以在任务停止后单独重新合并（如果使用过 limit，合并时沿用）：

```bash
python evaluations/direct_qa/run_qwen35_mmlongbench_2gpu.py --merge-only
```

可用 `--extractor-base-url`、`--extractor-model` 指定独立抽取器；也支持原来的
`EXTRACTOR_*` 和 `CITEAGENT_EVALUATOR_*` 环境变量。为了单独对照分辨率，
建议先保持抽取器等其他设置一致，再做独立抽取器对照。

## 基于当前日志的吞吐对照

纯 GPU 运行日志显示每卡权重约 17.66 GiB、KV cache 约 4.3 GiB / 138353 tokens。
16 并发下 KV 多次接近 100%，并出现 Running 减少和 Waiting 增加；日志没有明确的
preemption 计数，不能仅据此断定重算次数。先停止旧入口，再以原结果目录测试：

```bash
python evaluations/direct_qa/run_qwen35_mmlongbench_2gpu.py --gpu-ids 1,2 --cpu-offload-gb 0 \
  --concurrency 4 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --mm-processor-cache-gb 1
```

这组参数是待验证的吞吐起点，不承诺更快。用相同一组题目对比 2、4、8 并发，
观察题/小时、KV 与排队；不同运行进度的文档长度差异可能很大，不能直接当作严格速度对比。
若增大 batch 后启动报告 KV 容量不足，可先退回 max-num-batched-tokens=1024。

`--mm-processor-cache-gb 1` 是每个 API/engine 进程的 CPU 多模态预处理缓存，
不是 CPU 权重 offload，也不是 GPU KV cache。两套独立服务的此类缓存总预算约 4 GiB，
还需加上其他 CPU 内存。它可减少重复页图预处理，但不能等同于跳过视觉编码器或 LLM 前向。
这项功能默认仍为 0，需显式开启。

每道题的输入顺序是相同文档页图在前、问题在后，很适合前缀复用。
目前服务未开启前缀缓存，同一文档的不同问题会重复计算相同前缀。
可选 `--enable-prefix-caching` 会同时传入 `--mamba-cache-mode align`：
本地 Qwen3.5 实现不支持 all 模式，align 在官方 recipe 中仍标为实验性。
建议先在新目录跑同文档的若干题，核对答案、日志中的 Prefix cache hit rate 和吞吐：

```bash
python evaluations/direct_qa/run_qwen35_mmlongbench_2gpu.py --gpu-ids 1,2 --limit 10 \
  --concurrency 4 --max-num-seqs 4 --mm-processor-cache-gb 1 \
  --enable-prefix-caching \
  --output-dir output/mmlongbench/qwen35_prefix_smoke
```

缓存收益依赖请求到达顺序、前缀是否已计算完成及缓存是否被淘汰；首次请求不保证命中。
两张卡有各自独立的缓存。当前仍按题号分片，同一文档可能在两张卡上各计算一遍，
按文档分片是后续可考虑的另一项调度优化，但需要重新定义分片配置。

另有 `--no-enforce-eager` 可尝试编译与 CUDA Graphs，
`--max-cudagraph-capture-size` 默认限为 max-num-seqs，避免捕获远超实际并发的大 batch。
这会增加启动时间和显存需求；在当前 24GB/128K 的紧张配置上可能启动失败，
需单独小样本验证，不默认开启。改变缓存或编译配置不改变请求页图、thinking 或输出预算，
仍可能因计算路径差异产生数值变化，正式比较需核对固定样本。

所有新参数均记录到 run_config/run_history，可在停止入口后调整并续跑。
当前已启动的进程不会因修改脚本改变配置，也不会被这次修改自动重启。

## 部署与运行

以下为原来的两卡 TP=2、每页限制 262144 像素的入口，保留供对照使用。

当前单张色块图测试服务的图片上限为 1，不能用于本文档评测。
下面使用专用服务（端口 8037、模型别名 `qwen35-mmlongbench`）。
如果原服务仍占用 GPU 1、2，需要先在原服务终端按 Ctrl-C 结束，或通过 GPU_IDS 指定另外两张空闲卡。

终端一：

```bash
conda activate qwen35
cd /media/data1/leizh/longdoc
bash evaluations/direct_qa/serve_qwen35_mmlongbench.sh
# 换卡示例：GPU_IDS=4,6 bash evaluations/direct_qa/serve_qwen35_mmlongbench.sh
```

服务启动完成后，终端二：

```bash
conda activate qwen35
cd /media/data1/leizh/longdoc
python evaluations/direct_qa/evaluate_qwen35_mmlongbench.py --dry-run
python evaluations/direct_qa/evaluate_qwen35_mmlongbench.py --limit 2
# 前两题跑通后评测全部；保留并跳过已完成的前两题。
python evaluations/direct_qa/evaluate_qwen35_mmlongbench.py
```

相同命令可断点续跑；已生成回答会立即保存，抽取失败时重试不会重新生成回答。
默认客户端并发 1，服务端调度上限 8；评测时可用 `--concurrency 16` 提交请求。
每题失败会记录错误，**不会自动截页或换成检索**。

客户端 `--concurrency` 只增加提交请求的数量。旧部署使用 `MAX_NUM_SEQS=1`，
客户端设置 16 时只有一条序列运行，短答案抽取也共用队列。
新默认上限 8 是吞吐测试起点；可重启服务使用下列配置，再根据吞吐、KV cache 使用量和抢占重算情况调整：

```bash
# 终端一，需先结束旧服务；修改脚本不会改变已经运行的服务。
MAX_NUM_SEQS=8 bash evaluations/direct_qa/serve_qwen35_mmlongbench.sh
# 终端二，原结果文件支持续跑。
python evaluations/direct_qa/evaluate_qwen35_mmlongbench.py --concurrency 16
```

长短文档耗时差异很大；提高 MAX_NUM_SEQS 会增加显存需求，不代表一定提速。
短文档可对比 8 与 16，若长文档出现频繁 preemption/recompute 或吞吐下降，可退回 4。
启动日志中的 `Maximum concurrency ... 3.90x` 是所有请求都占满 128K 的 KV 容量估计，
不是 max-num-seqs 的硬上限；实际同时运行数量由调度器和可用 KV cache 共同限制。

## 默认评测设置

- 归档数据：1,082 题、135 份文档、6,522 张页图；最长 468 页。
- 使用 `data/mmlongbench/images_1024` 中的全部页面。
- 视觉预处理：每页 `min_pixels=65536`、`max_pixels=262144`，保留长宽比例。
  **每页分辨率会降低**，并不等同于以归档 1024 长边原图分辨率进行视觉编码。
- 服务允许最多 512 张图片、上下文长度 131072、分块预填充 2048 tokens。
  实际能否容纳最长文档取决于 token 数和显存，必须以实际运行结果为准。
- 与归档旧脚本一致，回答默认关闭 thinking，输出预算 8192 tokens，temperature=0；抽取也关闭 thinking，预算 2048 tokens。
- 默认复用本地 Qwen3.5 服务抽取短答案；这与原 CiteAgent 的 `gpt-oss-120b` 抽取模型不同，结果报告需注明。
- 分辨率、thinking、提示词、抽取模型变化会影响结果，比较实验时应保持一致。

修改设置需要使用不同输出文件。例如：

```bash
python evaluations/direct_qa/evaluate_qwen35_mmlongbench.py --enable-thinking \
  --output output/mmlongbench/qwen35_thinking_comparison.json
```

此前旧版新入口误将 thinking 默认设为 true，已产生的 `qwen35_direct.json` 是开启思考的结果。
现在不带开关会将关闭思考的实验保存到 `qwen35_direct_no_thinking.json`，避免复用原有思考答案。
已有结果完整保留；如需继续原来的开启思考实验，使用 `--enable-thinking` 即可。

调整视觉分辨率时，应同步修改服务启动的 `MAX_PIXELS` 和评测的 `--max-pixels`，
并按实际视觉 token 数增大 `MAX_MODEL_LEN`；脚本不承诺每种分辨率都能容纳所有文档。

## 指定独立短答案抽取模型

可以直接通过参数或环境变量指定：

```bash
export EXTRACTOR_BASE_URL=http://YOUR_EXTRACTOR_HOST:PORT/v1
export EXTRACTOR_MODEL=gpt-oss-120b
# 如果抽取服务需要认证，另设置 EXTRACTOR_API_KEY；本地无认证服务可不设置。
python evaluations/direct_qa/evaluate_qwen35_mmlongbench.py \
  --output output/mmlongbench/qwen35_gptoss_extractor.json
```

也支持归档的 `CITEAGENT_EVALUATOR_BASE_URL/MODEL/API_KEY` 环境变量。
回答服务使用 `QWEN_BASE_URL`、`QWEN_MODEL`、`QWEN_API_KEY`（或 `API_KEY`）。
不内置密钥、不交互提示输入密钥、不将密钥保存到结果文件。

若需先生成，再用独立模型抽取：

```bash
python evaluations/direct_qa/evaluate_qwen35_mmlongbench.py --stage predict
python evaluations/direct_qa/evaluate_qwen35_mmlongbench.py --stage score \
  --extractor-base-url http://YOUR_EXTRACTOR_HOST:PORT/v1 \
  --extractor-model gpt-oss-120b
```

## 输出与分数

默认 `output/mmlongbench/qwen35_direct_no_thinking.json` 逐题保存原始样本、回答、thinking、
输入页数、短答案、官方 score、token 用量、结束原因和错误。

- `.meta.json`：数据校验和、模型、端点、分辨率、thinking、输出预算等续跑配置。
- `.summary.json`：完成量、失败量、ACC、F1、百分制分数及是否全部完成。
- `.txt`：官方口径的单页/跨页/不可回答题、证据来源、文档类型分层报告。

ACC 是官方逐题 score 的均值，部分字符串题采用 ANLS，可出现小数得分。
F1 按归档的 answerable/unanswerable 定义计算，不是普通 token F1。
错误和未完成题目按 0 分进入总体分母，避免因跳过失败样本而抬高分数。
只有 `evaluation_complete=true` 且 `complete=1082` 才表示全量评测完成。
`truncated_answers` 会单独报告输出达到 token 上限的回答数量。

## 滚动 PDF 调度与复现检查（2026-10-30）

`test_prefix_fix.sh` 现在默认每卡总并发 16、最多 4 份活跃 PDF、最多 2 个冷请求，
先完成每份 PDF 的首题**主推理**，随即放行后续题目；提取和评分不再阻塞该 PDF。
主推理与提取共享总并发额度，提取最多占 2 个位置。一份 PDF 的主推理全部结束后
立即补入新 PDF，不需要等待其他文档或该 PDF 的提取完成。
主推理失败的首题不会将文档标记为已预热；下一题会重新尝试预热。

原来的全量命令仍可直接使用：

```bash
bash evaluations/direct_qa/test_prefix_fix.sh \
  --limit 0 --concurrency 16 --max-num-seqs 16 --timeout 900
```

默认结果目录改为 `output/mmlongbench/qwen35_a100_doc_prefix_mm4_rolling_seed0`，
避免复用已经完成的旧实验。分片、提示词、输入分辨率、整页裁剪和评分算法保持原样。
新增 `temperature=0`、提取 `temperature=0`、`seed=0` 显式请求参数；服务 seed 也固定为 0。
这些参数与批次一致性开关写入元数据；改变它们需要新输出目录，旧结果不会被删除。
仅调整窗口、总并发、冷请求数、提取并发和缓存预算可以续跑；测速度则必须使用新目录。

| 参数 | 入口默认值 | 含义 |
|---|---:|---|
| `--document-window` | 4 | 活跃 PDF 上限；设为 1 可恢复原逐 PDF 调度 |
| `--warmup-concurrency` | 2 | 同时进行的首题冷请求上限 |
| `--extraction-concurrency` | 2 | 提取并发上限，计入总并发 |
| `--document-token-budget` | 350000 | 活跃文档估计前缀 token 总预算；0 禁用该约束 |
| `--page-content-cache-mb` | 512 | 每个客户端进程的已编码页图缓存 MiB 预算 |
| `--temperature` / `--extract-temperature` | 0 / 0 | 主推理、提取的温度 |
| `--seed` | 0 | 请求与本地服务的固定 seed |
| `--batch-invariant` | 关闭 | 实验性 vLLM 批次一致性，启用需用新目录 |

token 预算只用于客户端文档准入估计，不是 vLLM GPU KV 预留；一个文档超过预算时
允许其单独运行。长文档可能使实际活跃 PDF 数小于 4。客户端图片缓存按编码字节数
和窗口数限制（不等于进程 RSS 上限），相同页集的并发缺失只读图/编码一次；
超过单项预算的文档仍可推理，但不会驻留缓存。服务端 `mm-processor-cache-gb` 是独立参数。
直接调用 Python 入口时 `document-window` 仍默认为 1，保留其他调用方的原有行为。

温度 0 和固定 seed **不保证不同批次或缓存状态下逐字一致**。本地 vLLM 有
`VLLM_BATCH_INVARIANT` 实现，但 Qwen3.5 多图 + Mamba align 补丁组合需单独验证。
`run_replicas.py --batch-invariant` 会给所有本地服务设置该环境变量，并记录配置。
独立调用 evaluator 时，这个参数只记录连接服务的设置，需自行正确配置服务；
指定外部提取服务时，本入口也不能替它开启 batch invariance。

先在两个新目录重复跑同一批问题，随后再改并发做对比；例如：

```bash
bash evaluations/direct_qa/test_prefix_fix.sh --limit 256 --timeout 900 \
  --batch-invariant --output-dir output/mmlongbench/repro_a
bash evaluations/direct_qa/test_prefix_fix.sh --limit 256 --timeout 900 \
  --batch-invariant --output-dir output/mmlongbench/repro_b
python evaluations/direct_qa/compare_reproducibility.py \
  output/mmlongbench/repro_a/merged.json output/mmlongbench/repro_b/merged.json
```

对比工具按原始题号检查主回答、提取文本、短答案和分数，报告一致率与差异题号；
有输出差异返回 1，输入题集不匹配或评测未完整完成则报错，不会把缺失结果当成一致。
单次一致不代表所有形状和输入都确定；还应测试并发 1/16/32 和长短文档。
上例前 256 题用于快速检查；正式提速对照应另选包含超长文档的相同固定题集。
先比较 `--document-window 1` 与滚动 4 的总耗时，再单独调整并发 32、
`--max-num-batched-tokens 8192`，同时观察命中率、抢占、错误数和结果一致率。

本次已完成 CPU 模拟请求、调度和配置验证；当前执行环境不可访问 GPU，尚未实测
新调度的吞吐收益或批次一致性开关的模型兼容性，不预先声称达到确定性或提速倍数。
