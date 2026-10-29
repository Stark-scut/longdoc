# Qwen3.5-9B：MMLongBench-Doc 全文页图直接评测

找到的旧脚本：`aaai_commit_20260803/aaai_commit/scripts/evaluate_mmlongbench_direct.py`。
新入口 `evaluate_qwen35_mmlongbench.py` 基于该流程适配当前本地 Qwen3.5 + vLLM；
默认数据、抽取提示词及评分函数均来自该归档，不改动归档文件。

每道题：按数字页号读取对应 PDF 的**全部** `page_0.png ... page_N.png`，
在同一个 user 请求里发送全部页图和问题，获得自由回答；再请求模型提取短答案，
最后调用归档 `eval_score.py` 对照 GT 评分。
回答和抽取请求都不包含 GT、gold evidence、OCR、检索结果或工具调用。

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
