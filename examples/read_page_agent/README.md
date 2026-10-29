# 最小 Qwen3.5 read_page Agent

只用原始问题、总页数和一个 `read_page(pages=[...])` 工具。
模型自主选择页面，工具返回 PNG 页图，再继续思考、调用工具或回答。
不使用原 CiteAgent 的 Locator/Selector/Reasoner、OCR、检索器或答案抽取器。

## 先部署，再测试

终端一启动独立服务，默认在 GPU 1 上部署一个 BF16 Qwen3.5-9B（TP=1）。
这是小规模 Agent 功能测试；端口 8057，模型名 `qwen35-read-page-agent`。

```bash
conda activate qwen35
cd /media/data1/leizh/longdoc
GPU_ID=1 bash examples/read_page_agent/serve_single_gpu.sh
# 如果 GPU 1 被占用，指定另一张空闲卡，例如 GPU_ID=4。
```

需要出现服务启动完成的日志。原色块测试服务只有 1 张图片上限，普通全文评测服务
没有启用自动工具解析，不能直接拿来跑这个多轮测试。
`GPU_ID`、`PORT`、`MODEL_PATH`、`MAX_MODEL_LEN`、`MAX_IMAGES`、
`MAX_NUM_BATCHED_TOKENS`、`CPU_OFFLOAD_GB` 和 `VLLM_GPU_MEMORY_UTILIZATION`
可通过环境变量覆盖。兼容旧文件名入口 `compat/serve_qwen35_read_page_agent.sh`，
现在也默认单卡；`GPU_IDS=1` 可用，`GPU_IDS=1,2` 会报错。
服务前台运行，Ctrl-C 停止；启动命令与日志保存在 `output/read_page_agent/server_*/`。
使用本机 `/tmp` 缓存，避开之前 NFS 上的 Triton 缓存问题。

单卡版本默认不使用 CPU 权重 offload，保留 BF16 精度；不做量化。
每卡一份完整模型，避免张量并行的跨 GPU 通信，但吞吐改善需要实测。
24GB 单卡显存紧张，设置 eager、每次最多调度 1 条序列、分块预填充 512 tokens、
显存比例 0.95、上下文 65536，给默认小样本测试使用。参数本身不保证不会 OOM。
如果实际启动报告 KV cache 容量不足，可以先用 `MAX_MODEL_LEN=32768` 测试；
上下文变小也会限制后续多轮图片和思考的累计长度。
如果仍然显存不足，可显式设置 `CPU_OFFLOAD_GB=2`，但这会引入 CPU/GPU 传输，可能降低速度。

如需两张卡各有一个独立副本，在两个终端分别启动：

```bash
GPU_ID=1 PORT=8057 bash examples/read_page_agent/serve_single_gpu.sh
GPU_ID=2 PORT=8058 bash examples/read_page_agent/serve_single_gpu.sh
```

测试时用 `--base-url http://127.0.0.1:8058/v1` 选择第二个服务。
本次最小 Agent 测试默认只连接一个副本，四道题依次运行。

终端二运行：

```bash
conda activate qwen35
cd /media/data1/leizh/longdoc

# 先检查默认题目和页图，不发 API 请求。
python examples/read_page_agent/run.py --dry-run

# 单题验证：10 页文档，问题要求辨认示例图里的蓝色数字。
python examples/read_page_agent/run.py --sample-indices 281 --k 2 --max-rounds 8 --show-thinking

# 默认 4 题，依次运行，每题使用独立的会话。
python examples/read_page_agent/run.py --k 2 --max-rounds 8 --show-thinking
```

`--k` 是一次最多读取的页数，模型可以少读；工具接受显式页号列表。
API 使用真正的 `tools` 和 `tool_choice=auto`，由 vLLM 的 `qwen3_coder` 解析器产生
`tool_calls`，代码不从模型的普通文本里解析假工具调用。
每轮最多执行一个调用；非法参数以 `role=tool` 的错误结果反馈，允许下一轮修正。

## 默认样本

题号是本地 `samples.json` 中的 **0-based 索引**。

| 题号 | 文档页数 | 问题类型 |
| --- | ---: | --- |
| 281 | 10 | 图中的蓝色数字计数 |
| 283 | 10 | 系统里使用 LLM 的模块计数 |
| 289 | 9 | Amazon 相关数据集的 item 总数 |
| 291 | 9 | 数据集每个用户的平均词数 |

这些是手选的小规模功能题，不是随机抽样或 benchmark 成绩。
只有原始问题和文档总页数进入初始消息；GT、答案格式和证据页不发送给模型。
工具页号从 **1** 开始，`pages=[1]` 对应 `page_0.png`，
与页面内印刷页码不一定一致。模型需自主探索，初始请求不包含任何页图。

## 消息链与思考保留

所有轮次都沿同一个 messages 列表追加：

```text
system：工具使用与回答规则
user：文档总页数 + 原始问题
assistant：reasoning + tool_calls(read_page)
tool：tool_call_id + [页号文字, image_url, 页号文字, image_url, ...]
assistant：reasoning + tool_calls(read_page)
tool：tool_call_id + 页图
assistant：最终回答
```

工具图像直接放在关联到真实 `tool_call_id` 的 `role=tool` 消息里，
后续轮次保留此前的工具图像、assistant content、tool_calls 与服务返回的 reasoning。
当前 vLLM 0.25.1 接收 `reasoning` 并映射为模板使用的 `reasoning_content`；
旧接口响应中的 `reasoning_content` 也会被客户端读出并回传。

本地 Qwen3.5 模板在同一用户问题后保留工具循环的思考，遇到新用户问题则清理更早的思考。
因此把工具观察随意改成新 `user` 消息，确实可能改变模板中的思考保留边界。
但“API 的 role=tool”不等同于模型内部特殊角色：官方模板仍会把工具结果编码进
`<tool_response>` 包裹的 user token 段，这是模板的正常实现。
重要的是完整会话结构和模板边界，不是 token 串中是否出现 user。

旧归档 `citeagent/locator.py` 还会每轮 `_build_messages` 重建上下文，
所以不能只改一个 role 字段就断定修好了旧系统，也不能从这项功能测试断定旧分数下降的原因。
原归档代码在这次测试中不做修改。

## 参数、日志与结束条件

- `--max-rounds 8`：最多 8 次模型请求，包括最终回答轮；到上限就停，不额外强制生成答案。
- `--max-tokens 8192`：每轮 thinking 与正文共享的输出预算；`finish_reason=length` 记为未完成。
- `--show-thinking`：终端显示 API 返回的 reasoning；默认只显示长度，完整内容始终保存。
- `--base-url` / `--model`：可指定兼容的现有服务。
- `--output-dir`：指定空结果目录；已有测试记录时拒绝覆盖。

默认服务开启 thinking、reasoning parser 和 function calling，
上下文为 65536，最多 32 张累计历史图片。
页图来自 `images_1024`，用 `min_pixels=1024,max_pixels=1048576` 保留现有分辨率，
只进行模型要求的 32 像素对齐；客户端检查源图不会因上限而明显缩小。
重复读页也会增加历史中的图片数量。增加轮数和 k 可能超出图片或上下文上限，
此时记录 HTTP 错误，不自动缩图、删历史或改用 user 消息。

每轮会保存到 `output/read_page_agent/test_*/sample_XXXX.json`：

- 全部 assistant 响应，包括 reasoning、tool_calls、结束原因和 token 用量。
- 完整消息结构、工具参数、返回页面与本地图片路径、错误及最终回答。
- GT 与 answer_format 仅存本地供人工核对。

为避免同一图片在磁盘反复占用空间，日志里的 base64 数据 URL 用 hash 占位替代；
**实际 HTTP 请求中始终是真实图片数据**。`results.json` 汇总各题轨迹，
`summary.json` 统计完成题数、读图后完成题数与是否观察到 reasoning。
退出码 0 仅表示每题读图后正常回答，不表示回答与 GT 一致。
不自动评分或调用抽取器，避免把抽取器问题混进这个 Agent 功能测试。

参考：[官方 Qwen3.5 模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)、
[vLLM Qwen3.5 部署说明](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)。
