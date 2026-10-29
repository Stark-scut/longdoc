# Locator 直接选页的长文档 Agent

统一入口：从仓库根目录执行 `python -m agent`。本目录包含完整程序、提示词和官方评分工具，不依赖旧 Agent 或固定基线。

## 当前流程

```text
Locator → search / read_page → call_reasoner(pages=[...]) → Reasoner
```

- read_page(pages=[3,8])：成功返回的 tool 消息只含两页的页号和整页图，保持原图；没有标题、OCR、crop或包含它们的JSON。
- Locator 自己决定证据页；call_reasoner(pages=[3,8]) 将这两页交给Reasoner。没有Selector模块或隐藏选页请求。
- Reasoner 从后台获取这些指定页的整页图、OCR和可用crop。
- 每次调用明确替换证据集合。第二次只传[8]，就只用第8页；要保留旧页，需把它们一起传入。
- 证据页必须已经成功read_page读取。空列表、未读页、非法页号会作为可纠正的工具错误返回，不调用Reasoner模型。
- search 继续在OCR/caption上检索并返回简短摘要；初始大纲仍用于定位。这次精简针对read_page的返回内容。
- 页面图像与原生assistant/tool历史仍持续保留，当前没有历史压缩。


## Locator提前结束的处理（0.2.1）

Locator默认使用tool_choice=required。提示词大纲末尾再次明确其职责为取证和提交证据页。
若服务仍返回普通文本，该内容保存为locator_non_tool_responses，不直接作为raw_answer。

在原max_rounds和wall_seconds预算内，最多发出max_protocol_corrections次明确的控制器纠正（默认2次）：
已有已读页时，下一次Locator请求指定call_reasoner工具，页列表仍由Locator选择；
没有已读页时，先要求继续取证。框架不代选页面，不直接拿Locator答案判分。
Reasoner请求只包含原问题与选定页的真实证据，不把Locator草稿作为答案提示。

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
| tools.py | search、read_page、call_reasoner的参数与返回 |
| evidence.py / reasoner.py | 校验Locator指定的精确页集合，构造完整证据并作答 |
| evaluation.py / launcher.py | 独立快照、多GPU副本、并发、进度与中断续跑 |
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

每卡BF16 / TP=1，CPU offload=0；Prefix caching默认启用，MM processor cache默认1GiB。
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

## 输出与续跑

所有结果默认在output/agent/，源码目录不写题集或运行输出。
每次运行保存run_config.json、runtime.json、summary.json、merged.json、progress.json、
snapshot/agent完整代码、snapshot/inputs题集/OCR/caption/图像哈希、
server*/server.log、workers日志与心跳、samples逐题结果及attempt日志、assets真实图片。

本版本使用**协议schema=2**，配置记录locator-direct-evidence和image-only。
请用新输出目录；旧Selector运行的快照保留原状，不能混入本次续跑。
相同配置和输出目录重复运行可续跑，允许调整GPU数量/编号、端口和并发。
--retry-failed才重试已结束的失败题；QA成功但抽取失败时只重试抽取。
改变题集、提示词或语义配置进行新实验时使用新输出目录。

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
本次同时改变证据选择和Locator读页内容，准确率/速度变化不能归因于其中单个因素。
离线测试验证工程行为；真实模型小样本评测需另行启动，说明见 [本次变更](../docs/locator-direct-evidence.md)。
