# Qwen3.5 多图 Prefix cache 调度补丁（2026-10-29）

## 依据与限制

A100 上关闭 Prefix cache 能运行；开启后出现 `Running: 1, Waiting: 7`、
吞吐为 0、KV cache 约 2% 的停滞。关闭 CPU 图片预处理缓存消除了
`Expected a cached item` 断言，但没有消除停滞。

本地 vLLM `0.24.0+cu129` 保留了先限制 encoder 预算、再向下做 Mamba 块对齐的逻辑。
这与上游 [PR #51603](https://github.com/vllm-project/vllm/pull/51603) 修复的
多模态编码缓存争用时零 token 调度问题吻合。现场采样未正常落盘，因而尚不能
证明每一次停滞都来自此问题，也不能据此判断 A100 不支持 Prefix cache。

`patch_vllm_mamba_align.py` 针对当前安装文件做兼容适配：

- RUNNING 和 WAITING 两条路径都在 encoder 限制之前做 Mamba 对齐。
- 按绝对 token 终点对齐；encoder 截短造成块内起点时，先推进到下一个块边界。
- 保留当前版本共有前缀边界参数和正常的等待完整块预算策略。
- encoder embeddings 的范围包含 EAGLE lookahead 偏移，覆盖上游的配套修复。

这是适配当前版本的 backport，不是将其他版本的整个 scheduler 文件覆盖过来。
脚本用 SHA256 限定已核对的原始源码；重复应用不重复修改；出现额外改动时拒绝覆盖。
原文件备份为同目录 `scheduler.py.pre_51603`。`apply` / `restore` 会清除相应字节码缓存。
已启动的进程不会自动更新；新启动的共享 `.venv_grpo` vLLM 进程都会使用补丁。

## 检查与 GPU 验证

CPU 回归检查命令（不需要 GPU）：

```bash
cd /media/data5/leizh/longdoc
.venv_grpo/bin/python longdoc/tests/test_vllm_mamba_align_backport.py -v
```

9 项 CPU 检查已通过。测试执行实际调度文件的相关 AST 代码片段和原版
EncoderCacheManager，覆盖旧版零进度复现、两条队列路径的恢复推进、块边界、
编码缓存释放、EAGLE、文本解码、异步 KV 接收、共有前缀、80 组不同图片边界，
以及补丁重复应用和精确回退。它不是完整 scheduler 集成测试，也没有验证 GPU
kernel、模型答案质量或四卡吞吐。

**A100 实测更新：补丁应用后，原 `sample` 分片、图片预处理缓存 0 的配置已完成
前 256 题：`complete=256, errors=0, pending=0, evaluation_complete=true`，
无输入或答案截断。** 结果见
`output/mmlongbench/qwen35_a100_prefixfix_smoke/merged.summary.json`。
这是调度修复的实际验证；它不等于完整 1082 题已完成，也不代表缓存配置的严格准确率对照。

停止先前的评测和 py-spy，确认 GPU 已释放后，在 cn2 执行：

```bash
bash /media/data5/leizh/longdoc/longdoc/evaluations/direct_qa/test_prefix_fix.sh
```

当前脚本默认 GPU `0,1,2,3`，前 32 题，保持每卡并发 8、Prefix cache 开启，
图片预处理缓存改为 **4 GiB**，按 PDF 分片并预热首题，保存到新的
`output/mmlongbench/qwen35_a100_doc_prefix_mm4`。
按 PDF 分片时，前 32 题只涉及 5 份 PDF，某张卡可能没有分到题目；
这与全量数据固定分配、保持扩大 limit 后可续跑有关，不是服务故障。
建议用前 256 题比较四卡吞吐：

```bash
bash /media/data5/leizh/longdoc/longdoc/evaluations/direct_qa/test_prefix_fix.sh --limit 256 --timeout 300
```
诊断请求超时设为 180 秒；原客户端仍最多尝试 3 次，因此单请求最坏仍可能等待约 9 分钟。
不要用“服务已就绪”或“Prefix cache 有命中”代替完成验证。
先检查 `merged.summary.json` 的 `complete=32, errors=0, evaluation_complete=true`，
以及无持续的零吞吐停滞，再扩大到 256 题并与不缓存的固定样本答案对照。

```bash
# 原输出目录续测前 256 题
bash /media/data5/leizh/longdoc/longdoc/evaluations/direct_qa/test_prefix_fix.sh --limit 256

# 更长文档可能需要恢复正式请求超时；0 表示全量
bash /media/data5/leizh/longdoc/longdoc/evaluations/direct_qa/test_prefix_fix.sh --limit 0 --timeout 1800
```

可设置 `GPU_IDS`、`MODEL_PATH`、`PYTHON_BIN`、`OUTPUT_DIR`，额外 CLI 参数也可覆盖默认值。
脚本不使用 py-spy，避免再次因 tracer 残留导致 tracing stop。

## 回退

先停止使用此环境的 vLLM 任务，再恢复：

```bash
/media/data5/leizh/longdoc/.venv_grpo/bin/python \
  /media/data5/leizh/longdoc/longdoc/evaluations/direct_qa/patch_vllm_mamba_align.py restore
```

回退后原启动命令加 `--no-enable-prefix-caching` 可继续采用已确认能运行的方案。
再次执行 `test_prefix_fix.sh` 会重新应用补丁，因此回退后不要用该脚本启动普通评测。

## Prefix cache 命中率与按文档调度

原入口按 `题号 % GPU数` 分片，每张卡各有独立的模型和缓存。
当前数据的实际统计：

| 选题范围 | PDF 数 | 旧方案的 PDF × GPU 副本数 | 按 PDF 分片 |
| --- | ---: | ---: | ---: |
| 前 32 题 | 5 | 20 | 5 |
| 前 256 题 | 36 | 144 | 36 |
| 全量 1082 题 | 135 | 537 | 135 |

全量 135 份 PDF 中，132 份原先分散到全部 4 张卡，另外 3 份分散到 3 张卡。
同一 PDF 在多张卡各算一次冷前缀，会浪费重复计算；一次并发提交多题时，
首题尚未生成可复用前缀，其他题也可能冷启动。

`--shard-strategy document` 现在实现：

1. 根据**完整输入**按 PDF 绑定 GPU，以题目数量做确定性的贪心平衡。
   增大 `--limit` 不改变已有题目的 GPU 归属。全量每卡题数为 `269/271/271/271`；
   这是题量平衡，不是耗时平衡，长 PDF 仍可能造成尾部等待。
2. 每卡逐个 PDF 处理，默认 `--document-warmup` 先完成一个未生成回答的题目，
   然后最多以 `--concurrency` 并发处理该 PDF 的剩余题目，全部结束后再进入下一个 PDF。
3. 原始题号、题目、图片、提示词、答案预算和评分方法保持一致；输出按原题号合并。
   分片元数据与合并校验记录 document-v1，进度条使用真实分片题数。

`run_replicas.py` 本身仍默认 `sample`，兼容已有结果；测试脚本显式选择 `document`。
更换分片策略必须使用新目录，不能直接混入旧 sample 分片结果。
`--no-document-warmup` 可对照不预热首题的性能；它仍逐个 PDF 提交任务。
逐 PDF 排队对只有少数问题的文档可能降低并发利用率，所以不能仅凭命中率保证提速。

当前 `server2` 日志中约 17.9%～18.8% 的 Prefix cache rate 是 token 口径的近期
缓存查询命中比例，不是“相同 PDF 有多少题”的比例。首次冷请求、不同问题后缀、
短答案抽取请求、缓存淘汰、整块对齐和并发时序都会影响它。
如果平均每份文档有 8 题，理想情况下第 1 题冷启动、其余 7 题复用，公共文档部分
命中上限约 87.5%，也不会天然是 100%。实际需比较相同题集的总耗时、题/小时、
prompt 吞吐、错误数与命中率。Prefix cache 主要减少 prefill，不会免去答案 decode。

## 图片预处理缓存设为 4 GiB

`--mm-processor-cache-gb 4` 已在测试脚本中配置。
这是 CPU 图片预处理缓存，与 GPU Prefix/KV cache 不同。
按当前每卡一个 API 进程和一个 engine 进程的四副本部署，缓存预算合计约
`4 GiB × (4 + 4) = 32 GiB` CPU 内存，另外还需要模型加载、图片和服务进程内存。
它不能增加 GPU KV 容量，也不等于完整跳过视觉编码。

之前的 `Expected a cached item` 是独立的图片缓存问题，Mamba 调度补丁没有修改
`multimodal/cache.py`。增大为 4 以及集中同 PDF 访问有望减少淘汰，但不保证修复
所有并发缓存问题。新的 `document + mm4` 配置目前完成 CPU 调度/合并测试与 dry-run，
仍需 A100 实测确认无 HTTP 500、所有题目完整完成，并检查固定样本答案。
如该断言再次出现，可暂以 `--mm-processor-cache-gb 0` 保留 Prefix cache 继续验证。

本轮本地验证：新增 5 项不依赖网络端口的文档路由、预热顺序、扩大 limit 续跑、
分片合并、错分片拒绝、进度与空分片检查通过；原测试中的 5 项可执行检查通过，
4 项本地 HTTP 服务集成测试因执行环境不允许创建 socket 而未运行。
256 题新配置 dry-run 通过；真实 GPU 缓存命中率与吞吐对比尚未执行。

官方参考：
[自动前缀缓存](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)、
[多模态缓存参数](https://docs.vllm.ai/en/latest/api/vllm/config/multimodal/)。

## py-spy 残留导致显存无法释放

本次排查还遇到 EngineCore 收到 TERM/KILL 后仍存在。实际线程状态是
`t (tracing stop)` / `ptrace_stop`，`TracerPid` 指向采样器的跟踪线程，
不是已经证实的 NVIDIA 驱动不可中断等待。
先查清跟踪器的进程号及身份，再终止属于自己的采样器，解除跟踪后清理 EngineCore。

```bash
# 将线程号替换为 /proc/<EngineCore PID>/status 中当前的 TracerPid
TRACE_TGID=$(awk '/^Tgid:/ {print $2}' /proc/<TracerPid>/status)
ps -p "$TRACE_TGID" -o user,pid,ppid,stat,args
# 确认属于本次 py-spy 后，再 kill -TERM "$TRACE_TGID"；必要时使用 KILL。
```

当前测试入口不使用 py-spy。信号清理问题与原始 Prefix cache 调度停滞应分别判断。

## 增大并发并运行全量

用户随后确认 `document + document-warmup + mm4` 的 Prefix cache 命中率和实际评测速度
均提高。当前保存的配置为每卡 `concurrency=8, max_num_seqs=8`。
例如一份 PDF 有 8 题：先单独完成第 1 题，再最多并发其余 7 题；后者可复用首题
留下的前缀，正常预热成功时不会 8 题同时冷启动。
缓存仍可能受淘汰、块边界、输出和实际截断页数影响，不能保证每题都 100% 命中。

全量 135 份 PDF 的题数中位数为 8、最多 18；106 份 PDF 不超过 9 题。
当前逐 PDF 排队策略下，把上限从 8 提高到 16 主要能增加其余 29 份 PDF 的并发。
实际并发还受当前文档剩余题数、服务调度和显存约束，四卡不是始终各运行 16 题。

停止旧任务后，可在同一 document 输出目录续跑全量并尝试每卡上限 16：

```bash
bash /media/data5/leizh/longdoc/longdoc/evaluations/direct_qa/test_prefix_fix.sh \
  --limit 0 --concurrency 16 --max-num-seqs 16 --timeout 900
```

这会保持 Prefix cache、首题预热和 mm4，跳过已完成题目；`--limit 0` 表示全量 1082 题。
仅提高客户端 concurrency 而不提高服务端 max-num-seqs，通常主要增加排队。
16 并发命令已通过全量 dry-run，真实 GPU 下的显存、吞吐收益仍需本次运行确认。
若吞吐下降或出现显存错误，可恢复两个并发参数为 8，沿用同一目录续跑。
