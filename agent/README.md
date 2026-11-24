# Locator 显式 Evidence memory 的长文档 Agent

统一入口：从仓库根目录执行 `python -m agent`。本目录包含完整程序、提示词和官方评分工具，不依赖旧 Agent 或固定基线。

## 当前流程

```text
Locator → search / read_page → store(pages=[...]) → Evidence memory
Locator → call_reasoner({}) → Reasoner读取全部memory页面 → answerable / insufficient
```

- read_page(pages=[3,8])：成功返回的 tool 消息只含两页的页号和整页图，保持原图；没有标题、OCR、crop或包含它们的JSON。
- Locator通过store(pages=[3,8])把已读且相关的页存入本题独立的memory；read_page不会自动入库。
- store默认追加、去重；不提供remove、clear、replace。重复存页不改变memory版本，新增页才增加版本。
- call_reasoner({})无参数，Reasoner从本次调用的memory快照获取全部已存页的整页图、OCR和可用crop。读取memory不改变其中的页面。
- Reasoner报告insufficient时保留已有证据，Locator继续search/read/store后再次调用。
- 空列表、未读页、非法页号不能store；空memory不能推理，均返回可纠正错误，不调用Reasoner模型。
- search只在OCR/caption上进行文本检索并返回简短摘要；本轮没有视觉/混合检索。
- 页面图像与原生assistant/tool历史仍持续保留，当前没有历史压缩。
- 新读页的图像观察须先交给Locator，下一轮才可store；store→call_reasoner可同批顺序执行。前一个工具失败后，其余调用返回dependency_failed，下一轮由Locator纠正。
- memory保存原始页面引用，不复制图片；不会因上下文不足自动丢页，服务端上下文/图像限制会如实记录失败。

例如store([3,8])后再store([12])，两次call_reasoner分别读取[3,8]和[3,8,12]。
完整约定和基线提交见[Evidence memory变更](../docs/evidence-memory.md)。


## Locator提前结束的处理（0.3.0）

Locator默认使用tool_choice=required。提示词大纲末尾再次明确其职责为取证和提交证据页。
若服务仍返回普通文本，该内容保存为locator_non_tool_responses，不直接作为raw_answer。

在原max_rounds和wall_seconds预算内，最多发出max_protocol_corrections次明确的控制器纠正（默认2次）：
memory非空时，下一次Locator请求指定无参数call_reasoner；
仅有已读页时，指定store，具体存哪些页仍由Locator选择；
没有已读页时，先要求继续取证。框架不代选、不自动store，不直接拿Locator答案判分。
Reasoner请求只包含原问题与memory页面的真实证据，不把Locator草稿作为答案提示。

protocol_corrections、protocol_correction_rounds及protocol_correction事件记录纠正过程；
所有实际请求仍计入成本。controller_fallback_calls=0表示没有框架直接代调Reasoner。
可在独立配置中设置locator_tool_choice=auto、max_protocol_corrections=0测试无纠正路径。
这些是语义配置，需新输出目录；旧快照不会自动采用新代码。

## 代码分工

| 模块 | 职责 |
| --- | --- |
| cli.py / config.py | 命令与配置校验 |
| loop.py | 原生工具循环、历史与停止原因 |
| client.py | thinking、HTTP、重试和JSON纠正 |
| documents.py | 页面、OCR、crop与检索；Locator和Reasoner使用不同渲染接口 |
| tools.py | search、read_page、store、call_reasoner的参数与返回 |
| evidence.py / reasoner.py | 追加去重、只读memory快照、构造完整证据并作答 |
| evaluation.py / launcher.py | 独立快照、多GPU副本、并发、进度与中断续跑 |
| scheduling.py | PDF固定归卡、卡内共享队列与滚动文档窗口 |
| logging.py | 请求响应、成本、图像资源与回放 |
| scoring.py / metrics/ | 短答案提取和官方ACC/F1；GT仅离线使用 |

## 启动

```bash
python -m pip install -r agent/requirements.txt

# 只检查命令和输入路径，不启动GPU、不写结果
python -m agent eval --config configs/agent/native.json \
  --gpu-ids 0,1 --indices 53,63 --dry-run

# 小样本：每卡一份完整模型
python -m agent eval --config configs/agent/native.json \
  --gpu-ids 0,1 --concurrency-per-gpu 1 \
  --indices 53,63,66,464,522 --output-dir output/agent/direct_pages_smoke_v2

# 全量另用新目录
python -m agent eval --config configs/agent/native.json \
  --gpu-ids 0,1 --concurrency-per-gpu 1 --output-dir output/agent/direct_pages_full
```

每卡BF16 / TP=1，CPU offload=0；Prefix caching默认启用，native.json的MM processor cache为3GiB（无配置时为1GiB）。
任意GPU编号与数量均可，例如 --gpu-ids 1,4,5；--max-num-seqs默认取每卡并发。
推理环境需另备兼容的vLLM、PyTorch、Transformers，客户端requirements不安装这些GPU组件。

完整数据仍在原 aaai_commit_20260803/aaai_commit/data/mmlongbench 目录；用 --data-dir 和 --model-path 覆盖新服务器路径。
逐项覆盖支持 --samples、--blocks、--images-dir、--crop-root、--captions。
角色仅有Locator、Reasoner和extractor；Locator默认thinking=true，其余false，可用 --no-thinking、--reasoner-thinking调整。

默认服务模型名longdoc-qwen35。已有服务用实际名字：

```bash
python -m agent doctor --base-url http://127.0.0.1:8067/v1 --model citeagent-qwen35
python -m agent eval --no-serve \
  --base-url http://127.0.0.1:8067/v1 --model citeagent-qwen35 \
  --indices 53,63 --output-dir output/agent/direct_pages_existing
```

多已有服务用 --no-serve --ports 8071,8072；结束时不会关闭外部服务。
不同URL的外部Reasoner/extractor不会被本地副本覆盖。

## 四卡吞吐配置

```bash
python -m agent eval --config configs/agent/native.json \
  --model-path /media/data5/leizh/models/Qwen3.5-9B \
  --data-dir /media/data5/leizh/longdoc/aaai_commit/data/mmlongbench \
  --gpu-ids 0,1,2,3 \
  --concurrency-per-gpu 16 --max-num-seqs 16 \
  --max-num-batched-tokens 8192 --max-model-len 65536 \
  --gpu-memory-utilization 0.95 --enable-prefix-caching \
  --mm-processor-cache-gb 3 \
  --shard-strategy document --document-window 4 --asset-mode reference \
  --output-dir output/agent/evidence_memory_4gpu_c16_mm3
```

默认使用document调度：按完整已选题集的每份PDF题数做贪心负载均衡，一份PDF的题固定分给一张GPU。
相同题集和GPU数量续跑时保留归卡关系；改变GPU数量时重新分配剩余题。
每卡的worker从同一文件队列领题，快的worker继续领下一题，不受原先静态题单限制。
--document-window默认4：在4份仍有待领题目的PDF之间轮流发题；一份PDF的题全部被领走后即补入下一份，
已经在执行的尾题可以继续运行。因此4不是GPU上所有在途PDF或实际缓存驻留量的硬上限。
不会额外执行首题预热，也不会等待某份PDF完整跑完才补题；已有服务用--ports时也采用同样调度。
--shard-strategy sample可切回原来的按题轮流分卡、静态worker题单，便于对照。

同PDF的大纲前缀与页图有更多机会命中本卡缓存，但不同问题和Agent轨迹不能共享全部KV。
每个worker最多保留4份文档检索索引（随document-window调整），编码图像另有64MiB的有界LRU，减少重复读图、编码与日志解码。
MM processor cache是服务端CPU缓存，与这里的图像编码缓存、GPU KV cache分别管理。
16并发是起始吞吐配置，实际速度需对比questions/hour、缓存命中、等待/抢占与温度；本次不修改Agent推理预算。
所有角色请求继续使用temperature=0和固定seed；并发变化仍可能影响数值与最终输出，不能保证逐次完全一致。

## 输出与续跑

所有结果默认在output/agent/，源码目录不写题集或运行输出。
每次运行保存run_config.json、runtime.json、summary.json、merged.json、progress.json、
snapshot/agent完整代码、snapshot/inputs题集/OCR/caption/图像哈希、
server*/server.log、workers日志与心跳、samples逐题结果及attempt日志。
默认--asset-mode reference：assets/只保存按图像哈希命名的小JSON，包含原图绝对路径、大小和SHA256，不另存完整图片；
LLM请求仍发送相同原图内容。回放会校验原图，源图移动、删除或内容变化后不能直接回放。
需要搬运日志时使用--asset-mode copy保存图片副本。无来源路径的外部data URL也会保留副本，以免丢失回放数据。
两种模式的旧日志均可回放；原来的输出目录与图片副本不会自动清理。

本版本使用**协议schema=3**，配置记录locator-evidence-memory、append-only和image-only。
请用新输出目录；schema=2直接选页及更早Selector运行不能混入本次续跑。
相同配置和输出目录重复运行可续跑，允许调整GPU数量/编号、端口和并发。
--retry-failed才重试已结束的失败题；QA成功但抽取失败时只重试抽取。
改变题集、提示词或语义配置进行新实验时使用新输出目录。
已有运行始终使用其冻结代码，继续运行不会自动启用memory。旧协议请使用原快照入口与原配置，或基线提交d749bba。
原直接选页基线标记为baseline/pre-evidence-memory-20261124，memory开发位于feat/locator-evidence-memory。
更改MM缓存配置同样会触发配置一致性校验。
逐题结果包含evidence_memory、evidence_pages、evidence_version和tool_counts；轮数仍指Locator请求轮次，store不单独增加模型请求。

按一次Ctrl+C保存已生成的QA并清理本次启动的进程组。重复信号不会打断清理。
技术失败、证据不足、轮数耗尽、截断和上下文超限分别记录；不把失败自动判为Not answerable。

```bash
python -m agent watch --output-dir output/agent/direct_pages_full
python -m agent merge --output-dir output/agent/direct_pages_full

# 分阶段时，两次都使用相同配置和输出目录
python -m agent eval --gpu-ids 0,1 --stage agent --output-dir output/agent/direct_pages_staged
python -m agent eval --gpu-ids 0,1 --stage score --output-dir output/agent/direct_pages_staged

python -m agent replay \
  --trace output/agent/direct_pages_full/samples/sample_0053/attempt_0001.jsonl \
  --sequence 2 --output output/agent/restored_request.json
```

sequence须对应实际llm_request；回放不发送请求。
日志保留reasoning、tool_calls、调用ID、结束原因和实际成本，不记录授权请求头。
未处理、未完成及技术失败按固定题集分母计零；官方列表等边界情况报明确评分错误。

## 与此前版本对照

带Selector、完整Locator读页观察的版本保留在chore/repository-layout分支（ef2d427）。
schema=2的直接选页/只读整图方案见[历史变更](../docs/locator-direct-evidence.md)。
schema=3在此基础上增加显式store与追加式memory；离线测试验证协议行为，真实模型效果需在新目录评测。
