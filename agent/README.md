# 独立长文档 Agent

本目录包含新 Agent 的全部程序、提示词和官方评分工具。不导入旧 Agent，不读取旧 baseline，不依赖旧目录中的 vendor。

## 分工

| 模块 | 职责 |
| --- | --- |
| cli.py / config.py | 统一命令、显式配置、输入校验 |
| loop.py | Locator 原生 assistant/tool 历史，预算与停止原因 |
| client.py | HTTP、thinking、重试、JSON 格式纠正及成本记录 |
| documents.py | 页面、OCR、crop、大纲与 BM25 检索 |
| tools.py | 工具 schema、参数校验和真实图像返回 |
| evidence.py / selector.py | 已读页、正式证据和版本；相关页选择 |
| reasoner.py | 基于正式证据回答或报告缺项 |
| scoring.py / metrics/ | 短答案抽取、官方 ACC/F1；GT 只用于离线评分 |
| evaluation.py | 代码与输入快照、隔离工作进程、逐题落盘 |
| launcher.py | 单卡完整模型副本、多卡并发、进度、清理和续跑 |
| logging.py | 原始请求响应、usage、工具观察、图片去重及回放 |

search/read_page → call_global_selector → call_reasoner；Reasoner 返回 answerable 后结束。
当前 read_page 仍为**整页图＋OCR＋可用crop**，参数仍为 pages。图像与消息完整保留，当前不做历史压缩。
只返回“整页图＋页号”的功能尚未加入，后续作为独立实验。

## 配置与运行

从仓库根目录执行：

```bash
python -m pip install -r agent/requirements.txt

python -m agent eval --config configs/agent/native.json \
  --gpu-ids 1,4,5 --concurrency-per-gpu 1 \
  --indices 53,63,66,464,522 --output-dir output/agent/smoke

# 全量评测用新输出目录，并去掉 --indices
python -m agent eval --config configs/agent/native.json \
  --gpu-ids 1,4,5 --concurrency-per-gpu 1 \
  --output-dir output/agent/native_full
```

每张卡完整 BF16 / TP=1，CPU offload=0；默认 Prefix caching 开启、MM processor cache=1 GiB、eager 开启。
`--concurrency-per-gpu` 是每副本同时执行的题数，默认1；`--max-num-seqs` 默认取该值。先用低并发验证，再对照吞吐。
图像按原文件发送，并预检 max_pixels 是否会触发分辨率缩减；不会为通过检查自动缩图。

输入覆盖参数：`--data-dir`、`--samples`、`--blocks`、`--images-dir`、`--crop-root`、`--captions`、`--model-path`。
角色默认 Locator thinking=true，其余false；可用 `--no-thinking` / `--reasoner-thinking` 覆盖。
角色 URL、模型、max_tokens、timeout、key_env 均可在 JSON 配置中分别指定。相对文件路径按仓库根目录解析。
仅设置 data_dir 会派生默认数据文件路径，逐文件参数再覆盖。配置不允许未知字段，防止参数拼错后静默失效。

服务别名默认 `longdoc-qwen35`。使用已部署的旧服务时，显式指定其实际名称：

```bash
python -m agent doctor --base-url http://127.0.0.1:8067/v1 --model citeagent-qwen35

python -m agent eval --no-serve \
  --base-url http://127.0.0.1:8067/v1 --model citeagent-qwen35 \
  --indices 53,63 --output-dir output/agent/existing_service
```

多个已有本地服务可用 `--no-serve --ports 8071,8072`；按各端口分配请求，结束时不会关闭外部服务。
不同 URL 的外部 extractor/Reasoner 不会被本地副本 URL 覆盖。
只检查命令用 `--dry-run`；只冻结代码与输入用 `--prepare-only`。

单独运行一道自定义问题：

```bash
python -m agent run --doc-id 2310.05634v2.pdf \
  --question "In figure 5, what is the color of the line that has no intersection with any other line?" \
  --base-url http://127.0.0.1:8067/v1 --model citeagent-qwen35 \
  --output-dir output/agent/question_63
```

## 输出、进度和恢复

```text
output/agent/native_full/
  run_config.json               题集、模型标识、冻结文件哈希与配置指纹
  runtime.json / run_history.jsonl
  run_status.json               starting/running/stopping/finished/interrupted/failed
  snapshot/agent/               本次完整独立代码，工作进程从这里运行
  snapshot/inputs/              选定题集、相关OCR/caption、图像哈希；页图不复制
  server0/server.log            每个模型副本的启动和服务日志
  workers/replica0_slot0.log    每个并发槽位日志
  workers/replica0_slot0.json   当前题目、轮次、角色与请求时间
  samples/sample_0053/result.json
  samples/sample_0053/attempt_0001.jsonl
  assets/                      按内容哈希去重的真实图片
  summary.json / merged.json / progress.json
```

终端显示进度、已判分/失败/未完成、吞吐、ETA 和活跃题目/轮次。
已完成结果立即落盘；答案先于抽取保存。相同命令复用相同输出目录即可续跑。
已结束的失败题默认不重跑；`--retry-failed` 才重试。QA成功但抽取失败时只重试抽取。
可以更换 GPU 编号/数量、端口和并发，已保存的题目不会因此重跑。
题集和语义配置变更必须使用新输出目录。旧实现的输出不能直接用于新 Agent 续跑。

按一次 Ctrl+C 停止：工作进程先中断，已保存QA保留；随后清理本次创建的vLLM进程组。重复 Ctrl+C、SIGTERM、SIGHUP 不会打断清理阶段。
最大轮数、模型输出截断、上下文超限和技术错误分别记录；技术失败不自动当作 Not answerable。

```bash
python -m agent watch --output-dir output/agent/native_full
python -m agent merge --output-dir output/agent/native_full

# 分阶段：第一阶段 agent 保存回答；第二阶段 all/score 完成提取
python -m agent eval --gpu-ids 0,1 --stage agent --output-dir output/agent/staged
python -m agent eval --gpu-ids 0,1 --stage score --output-dir output/agent/staged

python -m agent replay \
  --trace output/agent/native_full/samples/sample_0053/attempt_0001.jsonl \
  --sequence 2 --output output/agent/restored_request.json
```

回放 sequence 必须指向实际 llm_request；恢复请求但不发送。日志保存 reasoning、tool_calls、结束原因和费用；不记录授权请求头。
未处理、未完成和技术失败按固定题集分母计零；完成运行不等于全部成功判分。ACC 是官方 score 均值，F1 沿用官方汇总公式。

## 与旧系统的关系

独立实现保留同类能力，但提示词、BM25细节、工具返回布局与技术失败处理不是逐token等价迁移。
Selector 输入未变化时复用选择；不可恢复的结构错误直接记录真实原因，不再通过重复调用掩盖成轮数耗尽。
这些差异已列在 [迁移说明](../docs/layout-and-agent.md)，不可把新旧分数差直接归因为某一个变量。
