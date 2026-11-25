# 页级 Evidence memory（schema 3）

Locator现在用显式store维护证据。检索仅使用原来的OCR/caption文本检索；memory默认追加、去重，不提供remove、clear或replace。

## 基线与分支

改动前的工作已保存为三个提交：

- `5ab892b`：e2e的文档归卡、滚动调度及相关修复。
- `27c66b3`：Agent文档队列、图片路径日志和有界缓存。
- `d749bba`：既有实验分析和抽取问题方案。

整体基线标记：`baseline/pre-evidence-memory-20261124`，指向`d749bba`。
新协议开发分支：`feat/locator-evidence-memory`。
旧schema 2的实验继续使用原快照/原配置，或上述基线；新协议不能直接续跑旧目录。

## 四个工具

| 工具 | 参数 | 作用 |
| --- | --- | --- |
| search | query、可选top_k | 搜索OCR和caption，返回页号、分数及摘要 |
| read_page | pages | 返回所选页面的物理0-based索引和原整页图；不自动入库 |
| store | pages | 将成功读过的相关页追加到本题memory |
| call_reasoner | 空对象`{}` | 将memory全部页面交给Reasoner |

示例，所有索引均为物理0-based：

```text
read_page({"pages":[3,8]})
store({"pages":[3,8]})      → memory=[3,8], version=1
call_reasoner({})           → Reasoner使用[3,8]，返回insufficient与missing
search({"query":"缺少的事实"})
read_page({"pages":[12]})
store({"pages":[12,8]})     → added=[12], already_stored=[8]
                           → memory=[3,8,12], version=2
call_reasoner({})           → Reasoner使用[3,8,12]
```

store的返回包含`added_pages`、`already_stored_pages`、`memory_pages`和`memory_version`。重复存页不会增加version。`store({"pages":[]})`不是清空命令，会返回错误。带mode、remove、replace等额外参数同样被拒绝。

## 证据与状态规则

- 每次Agent.run为当前问题创建独立EvidenceMemory；题目间不共享memory，即使复用同一个Document或HTTP客户端。
- 未成功read的页不能store；单次store先校验全部页，非法项使整次store失败，不会部分入库。
- Memory保存Document页面引用，不复制图像、不保存Locator编造的摘要作为证据。
- Reasoner收到原问题，以及memory快照中所有页面的整页图、OCR和已有crop。已读但未store的页不进入Reasoner。
- 一次Reasoner请求固定页面快照与version；结果的`_input_pages`和`evidence_version`记录该次真实输入。
- Reasoner返回insufficient后不清空证据；后续store继续追加，下一次推理使用新的完整快照。
- 没有自动淘汰、删除、截断或摘要压缩。上下文/图像预算不足时保留memory并如实记录服务错误，不能静默省略页面。
- Memory与Locator的对话历史、服务端KV cache是不同状态；本次不裁剪对话历史。

## 调用顺序与预算

Locator必须先收到read_page的图像观察，新读页面才能在下一轮store；不能在首次read的同一批直接store，避免模型尚未看图就把页面入库。
同一条assistant消息可以返回store、call_reasoner两个调用，按返回顺序执行，并对每个tool_call_id生成观察。
前一个调用失败后，本批后续调用标为dependency_failed，不会在store失败后偷偷用旧memory推理；下一轮Locator可纠正。
已成功的前置store仍保留，只有单次store本身保证全部成功或全部失败。

max_rounds仍计Locator请求轮次，默认15；store是本地工具，不自己增加LLM请求，但Locator可以用一个额外轮次作store决策。
每题另记tool_counts，区分search/read_page/store/call_reasoner的实际执行次数；跳过的调用在tool_result日志中记录，不算执行次数。

若Locator提前输出普通答案：

- Memory非空：纠正为调用无参数call_reasoner。
- 只有已读页：纠正为store，由Locator选择相关页，不自动替它入库。
- 尚未读页：继续search/read，再store。

纠正次数、轮数和时间预算继续按配置约束；技术失败不转成语义不可回答。

## Locator状态栏

每个完整工具批次结束后，控制器将一个`<agent_status>`文本块附到最后一条真实tool消息的content末尾。
保留原tool_call_id、原结果和图片；不新增tool消息，也不为状态栏追加普通user消息。旧消息与旧状态栏保持原样。
首轮还没有工具结果，初始状态只加入初始system消息。最新状态栏代表当前状态，旧块是历史记录。

格式包含Round Budget、Document State、Evidence Memory、Tool Statistics、Reasoner Feedback和Recent Errors。
第11轮结束后将显示第12/15轮、已完成11轮、含下一次请求还剩4轮；终止时剩余轮数为0，不显示16/15。
页号范围按真实页面索引压缩，读过未存的页由集合差计算；该差集不意味着全部相关或必须入库。
Reasoner反馈保留上次实际评估的memory版本，便于识别其是否落后于当前memory。
工具计数包含失败尝试，不包含跳过调用；Recent Errors包含刚结束批次的全部错误，包括前序错误和依赖跳过。

原始tool_result事件仍保留原观察，新增agent_status事件记录增强后的最终消息，history_checkpoint和后续llm_request包含实际输入。
此改动不增加模型调用、不自动store、不强制调度，也不改变15轮/输出token/上下文预算。
既有提前输出普通答案时的user协议纠正分支未变；它与状态栏分开处理，不伪造工具结果。
请使用新输出目录验证状态栏，已有运行的冻结快照不会自动启用此改动。

## 结果与兼容性

`agent`版本0.3.0。配置使用：

```json
{
  "schema": 3,
  "pipeline": "locator-evidence-memory",
  "evidence_memory_mode": "append-only",
  "locator_read_page_format": "image-only"
}
```

逐题结果增加：

```json
{
  "evidence_memory": {"pages": [3,8,12], "version": 2, "mode": "append-only"},
  "tool_counts": {"read_page": 2, "store": 2, "call_reasoner": 2}
}
```

既有evidence_pages/evidence_version仍保留，含义改为本题当前memory的全部页面/版本。
保存的run_config顶层schema与配置均为3；旧协议由校验明确拒绝，避免新worker读取旧协议快照。

本轮未实施此前讨论的抽取协议、列表答案兼容、页码映射或thinking预算修复；这些可以独立提交和对照。

## 运行

在已验证可启动的vLLM环境里，先用新目录做小样本；不要使用旧a100_4gpu_direct_pages目录：

```bash
cd /media/data5/leizh/longdoc/longdoc
python -m agent eval --config configs/agent/native.json \
  --model-path /media/data5/leizh/models/Qwen3.5-9B \
  --data-dir /media/data5/leizh/longdoc/aaai_commit/data/mmlongbench \
  --gpu-ids 0 --indices 0,58,607 --concurrency-per-gpu 1 \
  --max-num-seqs 1 --max-num-batched-tokens 8192 \
  --mm-processor-cache-gb 3 \
  --output-dir output/agent/evidence_memory_smoke_v3
```

四卡16并发、MM cache 3GiB的完整命令见[Agent README](../agent/README.md#四卡吞吐配置)。
加--dry-run只检查配置、输入路径和调度计划，不启动GPU、不写实验结果。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests \
  /media/data5/leizh/longdoc/.venv_grpo/bin/python -B -m unittest \
  test_agent.Contracts test_agent_performance test_evidence_memory test_agent_status -v
```

测试覆盖追加去重、版本、未读页/非法页/原子store、题目隔离、不可变推理快照、无参数接口、同批依赖失败、
read只返回图与页号、文本检索、纠正分支、上下文失败不丢页、快照/配置隔离，以及冻结worker的独立子进程运行。
HTTP模型响应采用MockTransport，测试不需要GPU，也不将其结果当作真实模型准确率。
