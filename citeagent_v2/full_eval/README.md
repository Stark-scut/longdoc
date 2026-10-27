# 自选 GPU 的 native Agent 全量评测

入口：/media/data1/leizh/longdoc/citeagent_v2/full_eval/run_native_4gpu.py

按 --gpu-ids 指定的数量部署副本，每张 GPU 各部署一份完整 Qwen3.5-9B：BF16、TP=1、CPU offload=0。只评测修复后的 native 模式，默认跑 MMLongBench 全部 1082 题。原仓库、现有 citeagent_v2 核心文件、四题对照结果均不修改。

## 指定任意卡号和数量

入口保留 run_native_4gpu.py 文件名，现在支持单卡、双卡、三卡等任意正数副本，卡号无需连续。例如 GPU 1、4、5 各部署一个完整模型：

~~~bash
python /media/data1/leizh/longdoc/citeagent_v2/full_eval/run_native_4gpu.py \
  --gpu-ids 1,4,5 \
  --output-dir /media/data1/leizh/longdoc/citeagent_v2/runs/native_multi_gpu
~~~

默认端口按顺序分配：GPU 1 → 8071，GPU 4 → 8072，GPU 5 → 8073。可用 --ports 9071,9072,9073 自定义，数量必须与 GPU 相同。默认每卡并发 1，三卡同时评测 3 题；1082 题按原始题号 % 3 划分为 361、361、360 题。

新建的运行使用共享的不可变基线，GPU 分工保存在 runtime.json。中断后保持相同输出目录，可更换卡号或数量，例如改用 --gpu-ids 2,6；已完成题目跳过，剩余题目重新分配，已保存的 QA 不会因换卡而重跑。请先停止前一次启动，同一个输出目录仅允许一个启动器运行。原来已经冻结的四分片旧快照仍可按原卡数续跑；若需改变旧快照的卡数，请使用新输出目录，避免修改旧结果与冻结代码。

下文保留四卡示例。双卡只需改为 --gpu-ids 1,2；所有进度、日志、watch 与中断续跑功能同样有效。--output-dir 默认仍为 runs/native_4gpu，建议按上面的命令显式指定自己的评测目录，并始终复用它。

## 四卡启动示例

在 qwen35 环境中，指定四张空闲卡：

~~~bash
conda activate qwen35

python /media/data1/leizh/longdoc/citeagent_v2/full_eval/run_native_4gpu.py \
  --gpu-ids 1,2,4,6 \
  --output-dir /media/data1/leizh/longdoc/citeagent_v2/runs/native_4gpu
~~~

1,2,4,6 是示例卡号，按实际空闲情况替换。默认四个端口为 8071、8072、8073、8074。启动前检查四张卡的显存、利用率和端口；已有任务或服务不会自动停止。四个模型都启动成功后，才提交评测任务。

每张卡上的 Locator、Selector、Reasoner、抽取器共享同一个模型副本。四个分片同时评测，每个分片按原始题号 % 4 固定划分，默认每卡同时处理 1 道题，总共 4 道；不是四卡共同部署一个模型。

## 进度与日志

终端总进度包含已保存结果和本次新增结果，显示已判分量、技术失败量、预算耗尽量、每卡完成量、当前题号、轮次、组件和等待秒数。ETA 只按本次新增处理量估计，刚启动和文档长度变化时会波动。

~~~text
Native Agent ... 128/1082 [已判分120 技术失败5 预算/未完成3 |
G1 32/271 #129 r4:locator 20s | ... | ...题/h ETA ...]
~~~

“已处理”包括成功、技术失败和预算耗尽，不等于答对。最终 ACC/F1 按所有选中题目计算，失败与未完成按零分计入，同时单独报告覆盖率。

另开终端查看现有运行：

~~~bash
python /media/data1/leizh/longdoc/citeagent_v2/full_eval/run_native_4gpu.py \
  --output-dir /media/data1/leizh/longdoc/citeagent_v2/runs/native_4gpu \
  --watch
~~~

~~~text
runs/native_4gpu/
  run_config.json                 固定数据/语义配置与代码指纹
  runtime.json / run_history.jsonl 本次参数、分工与运行历史
  run_status.json                 starting/running/stopping/finished/interrupted/failed
  progress.json                   持续刷新的进度
  summary.json                    最终或中断时的 ACC/F1、覆盖率、失败统计
  merged.json                     全部题目的答案、GT、短答案及评分
  snapshot/                       本次实现、题集、OCR、模板、评分脚本和资源哈希
  server0/server.log              GPU 副本服务日志；server1～3 同理
  server0/launch.json              实际部署命令、GPU、端口、PID
  workers/shard0_slot0.log         每题完成日志
  workers/shard0_slot0.json        当前题目/轮次/组件心跳
  samples/sample_0281/result.json  原始推理答案、执行状态、短答案与评分
  samples/sample_0281/attempt_*.jsonl
  samples/sample_0281/assets       指向共享图片资源目录
  assets/                         所有样本共用的图片字节，按内容去重
~~~

每次请求和返回都保存，包括 reasoning、tool_calls、tool_call_id、finish_reason、usage 和真实图片资源引用。心跳在发起请求前更新，因此长时间等待时仍能知道卡在哪个组件。

## 中断与续跑

在启动终端按一次 Ctrl+C。脚本会通知工作进程保存结果，然后清理本次启动的 vLLM 进程组，包括 API 前端和 GPU 子进程。工作进程宽限 8 秒、服务宽限 15 秒，未退出时强制结束对应进程组。清理期间重复的 Ctrl+C、SIGTERM、SIGHUP 不会再次打断清理；关闭终端引起的 SIGHUP 也进入退出流程。run_status.json 先写入 stopping，清理结束后写 interrupted/finished/failed；清理错误单独保存。不会清理其他已有服务。

重新运行相同启动命令即可：

- 完成并判分的题目直接跳过。
- 原始答案已保存，但抽取/评分还没完成：只继续抽取评分。
- 中断时正在运行的 Agent 题目：重新执行这一题，旧轨迹保留。
- 已明确失败的题目默认保留结果，避免每次续跑重复等待同一失败。
- 使用 --retry-failed 再尝试技术失败、预算耗尽或抽取失败的题目；抽取失败但 QA 成功的题不会重新运行 Agent。

~~~bash
python /media/data1/leizh/longdoc/citeagent_v2/full_eval/run_native_4gpu.py \
  --gpu-ids 1,2,4,6 \
  --output-dir /media/data1/leizh/longdoc/citeagent_v2/runs/native_4gpu \
  --retry-failed
~~~

续跑以“题目”和“已完成的 QA 阶段”为单位，不承诺恢复被中断题目的某一条内部 reasoning。模型在请求中始终保留完整历史，没有增加压缩或截页策略。

首次准备输入时若被中断，也可以运行同一命令；脚本仅重建自己标记的未完成快照，不删除已有题目结果。

## 默认参数与调整

| 参数 | 默认值 | 含义 |
|---|---|---|
| --concurrency-per-gpu / --concurrency | 1 | 每卡同时评测题数，使用独立进程；总数为 GPU 数量 × concurrency |
| --enable-prefix-caching | 开启 | 默认启用 Prefix 缓存；--no-enable-prefix-caching 可关闭 |
| --mm-processor-cache-gb | 1 GiB | 默认启用 MM 预处理缓存；0 可关闭 |
| --max-num-seqs | 等于 concurrency | 每个 vLLM 的调度上限 |
| --max-num-batched-tokens | 2048 | 每步 token 预算 |
| --max-rounds | 15 | 全量 Agent 每题最多 Locator 请求轮数；四题实验原来是 8 |
| --read-page-max | 3 | 每次 read_page 最多读取页数 |
| --thinking | 开启 | Locator 思考开关，可用 --no-thinking 关闭 |
| --reasoner-thinking | 关闭 | Reasoner 思考开关 |
| --timeout | 600 秒 | 单次 HTTP 超时；比四题实验的 120 秒更长 |
| --wall-seconds | 3600 秒 | 单题 Agent 软时间预算；在请求之间检查 |
| --locator-max-tokens | 8192 | Locator 输出预算 |
| --reasoner-max-tokens | 8192 | Reasoner 输出预算 |
| --selector-max-tokens | 4096 | Selector 输出预算 |
| --extract-max-tokens | 2048 | 抽取输出预算 |
| --max-model-len | 65536 | 每个单卡服务上下文上限 |
| --max-images | 128 | 单个累计历史请求的图片上限 |
| --max-pixels | 1048576 | 保留 images_1024 页图，只做模型要求的对齐 |
| --progress-interval | 5 秒 | 进度刷新间隔 |

默认开启 prefix caching 和 1 GiB/进程的 MM processor cache，保持 eager。默认值变化需重启 vLLM 后生效；同时开启逐请求 prompt_tokens_details.cached_tokens，缓存开关记录在 runtime.json、run_history.jsonl 和实际服务部署命令中。现有输出快照不改动，工作进程的 Agent 配置仍按冻结基线执行。上下文或图片数量超限会记录错误，不会静默删历史、缩页或伪造答案。Reasoner 成功返回 answerable 后由框架结束，流程完成和答案正确分开统计。

每卡并发参数 --concurrency-per-gpu 与原 --concurrency 等价。默认仍为每卡 1 题，可显式设为 4；未指定 --max-num-seqs 时，vLLM 自动使用相同值，无需重复指定。显式 --max-num-seqs 不能小于每卡并发。以两张卡为例：

~~~bash
python /media/data1/leizh/longdoc/citeagent_v2/full_eval/run_native_4gpu.py \
  --gpu-ids 0,1 \
  --output-dir /media/data1/leizh/longdoc/citeagent_v2/runs/native_multi_gpu \
  --concurrency-per-gpu 4
~~~

两个缓存默认开启，上述命令每卡最多同时评测 4 题，总共 8 题，vLLM 每副本 max_num_seqs=4。runtime.json、run_history.jsonl 记录每卡并发、调度上限和缓存配置；server*/launch.json 记录实际服务命令。中断旧启动器并等待清理结束后，用原输出目录启动即可续跑。

并发增加能否提升吞吐需看实际题目耗时、排队、抢占与缓存占用，GPU 利用率和 KV 占用率本身不是优化目标。也可以每卡 2 题：

~~~bash
python /media/data1/leizh/longdoc/citeagent_v2/full_eval/run_native_4gpu.py \
  --gpu-ids 1,2,4,6 --concurrency 2 --max-num-seqs 2 \
  --output-dir /media/data1/leizh/longdoc/citeagent_v2/runs/native_4gpu
~~~

并发使用独立 Python 进程，隔离原代码的模块级客户端和状态，避免线程共享配置。可以在续跑时更换 GPU、端口、并发和超时时间，并记录运行历史；题集、图像、thinking、轮次、模型等改变时应指定新输出目录。

Selector 和抽取器固定关闭 thinking；可以在新运行时指定独立的抽取服务：

~~~bash
... --extractor-base-url http://HOST:PORT/v1 --extractor-model gpt-oss-120b
~~~

密钥从 CITEAGENT_EXTRACTOR_API_KEY 或 EXTRACTOR_API_KEY 读取，不存入配置/日志。默认仍由本卡 Qwen3.5 抽取，与四题新模式一致。

## 先检查、小样本与分阶段

只读取输入并打印计划，不写结果、不启动服务：

~~~bash
python /media/data1/leizh/longdoc/citeagent_v2/full_eval/run_native_4gpu.py \
  --gpu-ids 1,2,4,6 --dry-run
~~~

先跑前 8 题，使用独立输出目录：

~~~bash
python /media/data1/leizh/longdoc/citeagent_v2/full_eval/run_native_4gpu.py \
  --gpu-ids 1,2,4,6 --limit 8 \
  --output-dir /media/data1/leizh/longdoc/citeagent_v2/runs/native_4gpu_smoke
~~~

只准备输入快照，不部署：

~~~bash
python /media/data1/leizh/longdoc/citeagent_v2/full_eval/run_native_4gpu.py \
  --prepare-only \
  --output-dir /media/data1/leizh/longdoc/citeagent_v2/runs/native_4gpu
~~~

默认 --stage all 包含 QA、抽取和评分，也可先 --stage agent，之后同目录运行 --stage score。仅合并已保存结果时使用 --merge-only，不启动模型。

若已经手动启动四个兼容服务，可以传 --no-serve --ports 8071,8072,8073,8074。服务需提供 citeagent-qwen35 名称、工具解析器和对应的上下文/多图配置；脚本不会停止这些外部服务。

## 冻结和评分限制

工作进程执行 output/snapshot 中的实现，后续修改工作区代码不会混入这次续跑。小体积输入、OCR、caption、模板和评分脚本复制保存；大体积图片/crop 仅保存路径和 SHA256，每个工作进程首次使用某文档时校验。原始图片/crop 目录在评测期间应保留。模型元数据使用 SHA256，权重分片只记录文件名、大小和修改时间，不读取全部权重作哈希。

GT 只用于离线评分，Agent / Selector / Reasoner / 抽取请求不包含 GT 或 gold 证据页。所有题目都计入分母；技术失败、无有效答案和预算耗尽按零分计，不能自动获得不可回答题的分数。

summary.json 中 run_finished=true 表示选定题目均已处理（包含失败），evaluation_complete=true 才表示全部题目都成功判分。不能把流程完成率当作 QA 准确率。

## 测试

~~~bash
python /media/data1/leizh/longdoc/citeagent_v2/full_eval/tests.py
~~~

测试使用本地 HTTP 模拟服务和真实工作进程，覆盖单卡/双卡/三卡/四卡分片、变更副本数续跑、进度、已有结果跳过、抽取失败仅重抽、Ctrl+C 后保留 QA 并续跑，以及图片资源去重与历史工具链。模拟响应仅用于验证工程流程，不是实际模型得分。
