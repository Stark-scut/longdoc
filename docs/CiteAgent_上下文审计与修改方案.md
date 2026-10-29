# CiteAgent 上下文审计与修改方案

日期：2026-10-23  
审计目录：/media/data1/leizh/longdoc/aaai_commit_20260803/aaai_commit/citeagent  
参考材料：[Agent规范.pdf](/media/data1/leizh/longdoc/docs/references/Agent规范.pdf)，共 1 页。
交付范围：代码审查、离线模板验证和修改设计；本次不修改任何代码，不启动模型或评测。

## 1. 核心结论

**你的判断有依据。旧 Locator 的工具结果回传方式需要修改，而且需要同时修改消息历史管理，不能只把 user 换成 tool。**

当前代码存在三个相互关联的问题：

1. 模型真实发出了 function call，但程序将执行结果——包括页面图片、检索结果、Reasoner 判定——合并成普通 user 消息，没有通过 tool_call_id 与调用对应。
2. 每轮都重新构造 messages，没有把上一轮原始 assistant 消息及其 tool_calls、reasoning 接入后续请求。
3. 页面图像、检索返回和动作历史经过不同程度的删除、摘要或替换。程序保存过某页，并不代表模型下一轮还能看到该页的图像。

因此，它实现的是“根据外部状态重新构造下一次决策输入”，而不是完整延续的原生工具对话。这种状态驱动设计本身可以成立，但当前实现依赖未明确验证的信息压缩策略，也没有保留原生工具调用关系。

**不能据此断言：旧系统的所有错误都由这个问题造成，或者修好后准确率一定提高多少。** 需要通过固定条件的对照实验评估。你提供的 PDF 同样没有要求所有 reasoning 永久保留；它强调的是信息是否可重建，以及实验验证。

优先方案：保留现有 Locator → 工具 → Selector / Reasoner 分工，先让同一道题形成完整的 assistant → tool → assistant 链路，再设计有记录、可回查的上下文压缩。

## 2. PDF 中的原则应该怎样理解

PDF 图 1-3 区分系统指令、工具定义、思考过程、历史记录、工具结果。对本项目最有指导意义的是：

| 组件 | PDF 的含义 | 对 CiteAgent 的要求 |
| --- | --- | --- |
| 系统指令 | 定义角色和任务约束 | 保留稳定的 Locator 职责、证据规则、工具使用约定 |
| 工具定义 | 决定 Agent 能采取什么行动 | 每次请求正确传入 tools，定义与实际执行一致 |
| 工具结果 | 告诉 Agent 执行后实际发生了什么 | 真实返回页面、检索结果及错误，不能只留下“调用成功” |
| 历史记录 | 帮助避免重复操作和重复犯错 | 保存必要的动作、观察、结论及失败记录 |
| 思考过程 | 记录行动理由；部分信息可能可由观察重建 | 对 Qwen3.5 先建立能正确续接 reasoning 的基线，再做保留 / 去除对照 |

需要避免两个误读：

- PDF 是上下文设计和消融实验的讨论，不能当成所有模型、所有 API 都遵循的强制协议规范。
- “没有完整 reasoning”与“没有工具观察或动作历史”不是同一类问题。即使关闭 thinking，工具调用关系、图片观察和历史记录仍然需要正确处理。

旧代码并非完全没有工具结果和记忆：它保存了检索摘要、已读页记录、部分 OCR、证据 E 和 Reasoner 历史。问题在于这些信息经过了有损重建，而且来源角色、调用关系和部分图像没有保留。

## 3. 现有系统实际怎样运行

主要流程：

~~~text
原始问题
  ↓
Locator 决定 search / read_page / select / call_reasoner
  ↓
本地执行工具，并更新外部状态
  ├─ 检索历史
  ├─ 已读页面、OCR 和摘要
  ├─ 证据集合 E
  └─ Selector / Reasoner 结果
  ↓
丢开上一轮原始 messages，按上述状态重新生成：
  system + user(问题) + 多条 user(状态/证据/图片/最近工具结果)
  ↓
下一次 Locator 请求
~~~

关键代码：

- [locator.py：每轮上下文构造][locator-build]。
- [locator.py：每轮调用 _build_messages][locator-loop]。
- [locator.py：把所有工具结果组装成 user][locator-user-result]。
- [pipeline.py：最终答案取自 Reasoner][pipeline-decision]。

注意：Locator 不负责最终答案、由 Reasoner 统一裁决，是现有架构的明确设计。此次修复不需要顺便把它改成单模型直接回答。

## 4. 已确认的问题及影响

### 4.1 工具结果被当成普通 user 消息，调用与返回之间失去关联【P0】

位置：[locator.py：1270 起][locator-user-result]。

当前逻辑大致是：

~~~python
for tc in msg.tool_calls:
    result = dispatch(tc.function.name, args)
    # 所有调用的结果累积到同一个 next_tool_parts

last_tool_block = {"role": "user", "content": next_tool_parts}
~~~

存在以下问题：

- 页面图像来自 read_page，却被表现成用户新发送的图片。
- search、read_page、call_reasoner 等多个调用结果被合并，没有逐条 tool_call_id。
- 之后的上下文中没有对应的原始 assistant.tool_calls。
- 模型需要依靠文本描述推断“哪个结果属于哪个行动”，而不能延续完整的工具交互。

项目自身的 [llm.chat_with_tools 文档][llm-tools] 已要求调用方将结果以 role=tool 追加，但生产 Locator 没有按这个约定执行。

正确形式应当是：

~~~text
user：原始问题
assistant：reasoning + tool_calls(id=A, read_page)
tool：tool_call_id=A，页面标签 + 图片 + OCR
assistant：reasoning + tool_calls(id=B, search)
tool：tool_call_id=B，检索结果
assistant：下一步决策
~~~

每一个真实调用都应有对应的返回，包括参数错误、页面不存在、工具超时、无检索结果等情况。

### 4.2 每轮重建输入，原始 assistant 历史没有续接【P0】

位置：[locator.py：989 起][locator-build]、[1114 起][locator-loop]。

_build_messages 每次从 system 和原始问题开始，追加状态摘要、证据和最近工具结果。它没有读取之前真实的 assistant 消息。

因此，后续请求没有完整保留：

- assistant 原始 tool_calls 和调用 ID；
- assistant 的 reasoning；
- 模型之前明确提出的计划、假设、排除理由；
- 原始工具返回与后续决定之间的顺序关系。

外部状态可以补回一部分事实，但不能自动等价于完整的对话历史。例如，“读过第 4 页”不能替代“第 4 页的两组数值分别属于哪两个数据集，以及上轮为何选择其中一组”。

**只将 last_tool_block.role 改为 tool 会产生另一种不完整输入：存在工具返回，却不存在与它对应的 assistant 调用。修复必须覆盖整个消息循环。**

### 4.3 Qwen3.5 模板确实会区分工具观察和新用户问题【P0】

依据：本地 [Qwen3.5-9B/chat_template.jinja][qwen-query]。

模板会寻找最后一个真正的 user 问题：

- 普通 user 消息会更新 last_query_index。
- 正常 role=tool 不被视为新的用户问题。
- assistant 中的 reasoning_content，只有位于最后一个用户问题之后时才会按该模板保留。

相关代码：[查找最后用户问题][qwen-query]、[assistant reasoning 渲染][qwen-reasoning]、[tool 渲染][qwen-tool]。

本次实际执行了一个不调用模型的离线模板验证，使用两段人工占位 reasoning：

| 输入方式 | 第一段占位 reasoning | 第二段占位 reasoning | 工具返回块数量 |
| --- | --- | --- | --- |
| 完整 assistant → tool 历史 | 保留 | 保留 | 2 |
| 只把最后一条 tool 改为普通 user | 移除 | 移除 | 1 |
| 类似旧 Locator 的状态重建 | 不存在 | 不存在 | 0 |

这验证了两个独立机制：

1. 即便保留 assistant 历史，错误插入普通 user 观察，也可能让模板删除此前 reasoning。
2. 当前旧 Locator 在更早的应用层就没有把 assistant 历史传入，单纯修改模板无法补回。

这个实验验证的是模板行为，**没有测量准确率变化**。实际旧训练检查点是否使用同一模板，仍应通过当时服务参数和检查点内模板核对。

另一个容易误判的细节：本地模板会把 role=tool 渲染为带有 <tool_response> 标记的 user token 段。这属于模型模板的正常序列化，不能因此认为 API 层 tool 与普通 user 等价。

### 4.4 图片可见性与“已经读过”不一致【P1】

依据：[config.py：读页持久化配置][config-context]、[locator.py：图片与 E 构造][locator-build]。

当前默认值为：

| 配置 | 默认值 | 实际含义 |
| --- | --- | --- |
| LOCATOR_READ_CONTEXT | persistent_text | 后续重新展示预算内的已读 OCR 文本 |
| LOCATOR_READ_CONTEXT_MAX_CHARS | 24000 | 页面文本字符预算，超限页面退回摘要 |
| PERSIST_PAGE_IMAGE | 0 | 不额外持久附带最近一批整页图 |
| READ_PAGE_IMAGE | True | 本次 read_page 返回整页图 |
| LOCATOR_REPEAT_READ | allow | 允许重新读页 |

需要分清四层：

1. 磁盘上有页面图片。
2. Python 的 _read_page_history 中有页面记录。
3. Locator 本次请求里确实含有该图片。
4. Selector / Reasoner 的独立请求里确实含有该图片。

这些条件并不相同。

具体而言：

- 最近一次 read_page 的图片会出现在下一次请求的临时结果里，之后可能被新工具结果替换。
- 开启 PERSIST_PAGE_IMAGE 后，额外保留的也是“最近一次读页批次”，不是全部历史页面。
- Locator 对 E 的输入采用 to_prompt 加 crop_items，不能直接等同于 Reasoner 的整页图文 render_parts。整页被加入 E，不代表后续 Locator 一定能看到该整页图。
- OCR 不包含颜色、箭头、版面关系等全部视觉信息，不能默认替代原图。
- 文本预算按字符而非实际 token 计，也没有覆盖图像 token 和输出空间。

风险是模型记得“读过某页”，却已无法核对该页中的视觉细节。若采用 skip 策略，又没有图片回查机制，这种信息丢失更难恢复。

**这里的问题不是要求所有图片永久保留，而是图片何时删除、删除后保留什么、如何重新获取，都应有明确策略和记录。**

### 4.5 请求结束原因丢失，可能把截断误当成正常行为【P1】

位置：[llm.py：118 起][llm-tools]。

chat_with_tools 只返回 choices[0].message，没有把 finish_reason 和 usage 等信息提供给 Locator。

因此，上层无法可靠区分：

- 正常工具调用；
- 正常结束；
- 输出达到 token 上限；
- 其他异常结束情况。

其默认 max_tokens 为 4096，Locator 当前调用没有覆盖这个参数。thinking 会占用输出预算，是否发生截断应根据真实 finish_reason 判断。

建议返回完整 completion，或者返回统一结构：message、finish_reason、usage、request_id。对于 length，不把半段答案或不完整工具调用当作正常完成，也不执行未通过校验的参数。

### 4.6 Reasoner 的技术故障被转换成“证据不足”【P1】

位置：[reasoner.py：105 起][reasoner-errors]。

请求超时、解析失败等异常，当前会被转换为：

~~~json
{
  "status": "insufficient",
  "missing": [{"missing_claim": "推理器输出解析失败或超时"}],
  "_error": "..."
}
~~~

这把两个不同问题混在一起：

- 文档事实不足，需要继续检索。
- 推理服务或输出处理失败，需要技术重试或终止。

旧代码并非完全没有错误控制：[Locator 已实现错误缓存和预算][locator-reasoner]。但状态仍混用，普通评测轨迹还可能进一步省略 _error。

建议保留错误预算，同时增加明确的执行层状态：

~~~json
{
  "ok": false,
  "error": {
    "code": "reasoner_timeout",
    "retryable": true
  },
  "evidence_version": 3
}
~~~

只有服务成功完成判断后，结果中的语义状态才是 answerable 或 insufficient。技术错误不应自动生成“继续搜索原问题”的缺失事实。

### 4.7 催促消息、终止和兜底调用的语义需要明确【P1】

位置：[locator.py：1134 起][locator-nudge]、[pipeline.py：84 起][pipeline-fallback]。

当前逻辑在没有 tool_calls、Reasoner 又未判 answerable 时，会插入普通 user 催促继续调用。它可能重置问题边界，也没有保留刚才那条 assistant 的完整历史。

此外：

- max_rounds 计的是 Locator 模型请求轮数。
- 最后一轮如果刚调用 read_page，返回的图片可能没有下一次 Locator 请求消费。
- pipeline 还可能在 Locator 循环之外补调 Selector / Reasoner。

这些不一定都是错误，但必须清楚区分“模型正常完成”“预算耗尽”“框架额外补调”，不能全部记成 Agent 自主完成。

建议保持 Reasoner 裁决职责，并将终止要求放在初始系统规则中；必要时设计显式的提交动作，而不是反复伪造用户催促。对提前无工具结束，记录为相应控制状态，再按明确策略决定补调或终止。

补调结果应有 controller_fallback 标记，成本单独计入，不能伪造为模型主动发出的 tool_call。

### 4.8 现有日志不足以完整回放，修复后不能沿用原序列化器【P1】

位置：[locator.py：273 起][locator-dump]、[run_mmlongbench.py：66 起][eval-trace]。

当前有两类记录：

- 默认 trace：保存 assistant_text 和工具摘要，没有完整调用 ID、reasoning、finish_reason。
- 可选 LOCATOR_TRAJ_DUMP：能保存 teacher reasoning 和调用，但默认关闭。

还存在两个具体限制：

1. _dump_messages 只输出 role / content，图片只记数量，无法确定实际用了哪些图片；将来即便 messages 改成正确协议，该函数仍会丢掉历史 assistant.tool_calls、tool_call_id 和 reasoning。
2. compact_trace 会进一步压缩；Reasoner 只保留 status / answer / missing，_error 等诊断信息可能不进入最终紧凑轨迹。

因此“代码曾保存过 reasoning”与“后续请求保留了 reasoning”必须区分，当前前者只在特定日志开关下成立，后者在 Locator 主循环不成立。

### 4.9 旧训练输入可能依赖当前错误结构【迁移注意事项】

抽查了本地训练文件：

- [training_data/locator_train.jsonl][train-locator]；
- data/locator_train.jsonl。

两者首条样本均含 system、原始 user 问题、额外 user 状态块，再接 assistant 和 tool_call 训练记录。这说明至少部分训练材料确实采用旧的状态重建方式。

需要注意：

- 这里仅抽查了首条，不能据此判断所有训练样本或所有检查点。
- tool_call 是训练框架的数据表示，不能直接因为它不是 API 的 role=tool 就判定训练数据非法。
- 改为完整工具历史后，旧 Locator 检查点可能发生输入分布变化，应单独评测。
- 正确修复顺序是先建立基础 Qwen3.5 的协议基线，再验证旧检查点适配性；是否需要重新生成训练数据，由对照结果决定。

## 5. 哪些部分不应该误改

### 5.1 独立 Reasoner / Selector 的 user 输入可以是正常设计

[reasoner.reason][reasoner-entry] 是一个独立的子任务请求：接收问题与证据，返回充分性判断。以 system + user(图文证据) 提交给这个独立推理器是合理的。

需要改的是：它的返回交回 Locator 时，应成为对应 call_reasoner 的 tool 结果。

Selector 同理。不能把整个仓库所有包含图片的 user 消息机械替换成 tool。

### 5.2 外部证据记忆应保留

EvidenceMemory、已读页集合、检索缓存都具有实际价值，用于去重、证据选择和独立 Reasoner 输入。

建议明确区分：

- 对话记录：模型实际说过什么、调用过什么、看到了什么。
- 外部状态：程序保存的文档和证据。
- 请求视图：受预算限制，本次实际送给模型的内容。

外部状态可以补充对话，不能让其默默替代全部历史。

### 5.3 页号约定必须保持一致

旧系统内部使用 0-based 页号，数据集 evidence_pages 使用 1-based：[config.py][config-pages]。

现有简版 read_page 测试采用 1-based 对外工具页号。借鉴其消息循环时，必须增加明确映射，不能直接复制页号逻辑。

本次协议修复建议先保留旧工具的 0-based 约定；若未来改为对外 1-based，应作为独立迁移，同时修改工具说明、验证器、日志、评分转换和训练数据。

## 6. 建议的目标消息结构

同一道题初始化一次消息历史：

~~~json
[
  {
    "role": "system",
    "content": "Locator职责、工具规则、证据要求、终止条件"
  },
  {
    "role": "user",
    "content": "原始问题及必要的文档元数据"
  },
  {
    "role": "assistant",
    "content": "",
    "reasoning": "服务返回的reasoning；此处只是占位说明",
    "tool_calls": [
      {
        "id": "call_read_001",
        "type": "function",
        "function": {
          "name": "read_page",
          "arguments": "{\"pages\":[0,1]}"
        }
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_read_001",
    "content": [
      {"type": "text", "text": "page_index=0；这是第1个物理页面"},
      {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,<实际图片字节编码>"}
      },
      {"type": "text", "text": "该页OCR元素……"},
      {"type": "text", "text": "page_index=1；这是第2个物理页面"},
      {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,<实际图片字节编码>"}
      },
      {"type": "text", "text": "该页OCR元素……"}
    ]
  }
]
~~~

说明：

- 以上是当前本地 vLLM 兼容接口的目标结构示意，图片和 reasoning 都是占位说明，不能直接发送。
- 图片必须作为真正的 image_url 多模态 part 发送；把图片路径或整个图片对象转成普通 JSON 字符串，不等价于模型接收图片。
- 一次 read_page 返回 k 页，可以放在同一条 tool 消息中；每页标签、图片、OCR 应邻接。
- 一次 assistant 有多个 tool_calls，应为每个调用分别追加 tool 消息，不能把多个调用的结果混为一条。
- 保存真实的调用 ID、函数名和原始参数，不伪造、不重新编号。

当前 vLLM 接收 reasoning，并将其转换给模板使用；兼容旧返回时可读取 reasoning 或 reasoning_content，再由服务适配层按实际版本序列化。依据：[vLLM 请求字段规范化][vllm-protocol]、[模板消息转换][vllm-chat]。

不能假定 SDK 把响应对象原样转成字典后，每个响应字段都能直接作为下一轮请求字段。需要专门的 assistant 历史序列化函数。

## 7. 主循环修改方案：先做最小修复

保留现有工具实现和证据处理，重写 Locator 的会话组织：

~~~python
messages = initial_system_and_question()
state = initial_external_state()

for turn in range(max_model_turns):
    completion = request(messages, tools, explicit_thinking_config)
    save_response_and_usage(completion)

    if completion_is_truncated_or_failed(completion):
        record_incomplete_or_error()
        break

    assistant = serialize_assistant_for_history(completion.message)
    messages.append(assistant)

    if assistant_has_tool_calls(assistant):
        for call in assistant["tool_calls"]:
            # 成功和失败都形成对应的 observation。
            outcome = validate_and_execute(call, state)
            tool_message = serialize_tool_observation(call["id"], outcome)
            messages.append(tool_message)
        continue

    apply_explicit_terminal_policy(assistant, state)
    break
~~~

需要落实的规则：

1. 每道题拥有独立 messages 和 state，防止跨题串历史。
2. 每轮追加原始 assistant，再追加实际工具返回。
3. 工具参数 JSON 错误直接返回带调用 ID 的参数错误，不把无效 JSON 默默当成空对象执行。
4. 逐项验证页号、类型、范围和最多 k 页。不能把 true 当作页号 1，也不能默默把 1.9 转成 1。
5. 工具错误、无结果、部分页面失败均如实返回；缺失图片明确标记，不能宣称“已经看过图片”。
6. 第一版按当前顺序串行执行同轮工具，避免 read_page、select 和 call_reasoner 之间的状态依赖改变；并行工具执行留到之后单独验证。
7. 工具造成的 E 变化，随该工具返回提供新增证据、当前集合和 evidence_version，不再每轮追加多条伪用户状态。
8. call_reasoner 若内部自动触发 Selector，返回中应说明实际选中的证据和内部操作；内部调用日志与顶层模型工具调用分开记录。
9. thinking 开关、输出上限、超时和重试预算显式配置，并落盘实际生效值。
10. 避免记录完成后继续修改同一个消息字典，历史事件应保存为快照。

可以参考现有 [test_qwen35_read_page_agent.py][simple-agent] 的追加式消息循环和 tool 图片返回方式，但保留旧架构的 Selector / Reasoner 分工，并处理页号差异。

## 8. 上下文预算与图片保留方案

### 第一阶段：短文档完整历史基线

选择少量短文档、固定轮数和图像分辨率，保留完整 assistant / tool 历史及其图片，不做隐式删除。

目的：先证明协议、reasoning 续接、图片可见性和日志回放正确。这个阶段不能同时更换模型、重写工具、调整检索和压缩图片，否则难以归因。

### 第二阶段：显式压缩，而不是无限保留所有图片

完整历史会增加视觉 token、预填充成本和上下文压力，不适合作为无限长文档的最终策略。

建议按实际 token 预算管理：

- 为下一轮输出和潜在工具返回保留空间。
- 优先保留最近的完整工具交互及其图像。
- 保留与当前问题直接相关的证据页，必要时保留足以核对结论的局部图，但不能默认局部图替代全页。
- 历史搜索和无关 OCR 可压缩成结构化行动记录。
- 被移除的图像仍有稳定的页号、资源引用、摘要和重新读取路径。
- read_page 读取已读页面应支持重新回传图像，不因“过去读过”而永久拒绝。
- 明确记录每轮实际可见图片，而非仅记录累计 read_pages。

建议给页面维护两种状态：

~~~text
ever_read_pages：历史上读过
visible_image_pages：本次请求实际带图
~~~

两者不应混用。

压缩时不能留下孤立的 tool 返回或只有一半的调用对。也不能在对话中途随意追加 system 摘要：本地 Qwen3.5 模板要求 system 在开头，且不支持 system 消息含图。

较稳妥的后续方案是：在明确的边界建立新的压缩上下文，初始 system 保持任务规则，加入清楚标记的历史摘要和证据引用，再接续完整的近期交互。摘要包含的文档内容仍标记为数据。所有被保留的 tool 返回都必须有对应调用；发生上下文重建时记录 compaction 事件，并通过回归测试验证模型行为。

这是第一阶段之后的独立工程工作，不能作为协议修复的隐式附带变化。

## 9. 终止、故障与评分方案

建议将“执行是否正常”和“答案是什么”分开表示：

| 层面 | 建议记录 | 含义 |
| --- | --- | --- |
| 执行 | completed / incomplete / error | 是否正常走完流程 |
| 终止原因 | final_answer / max_rounds / context_limit / timeout / output_truncated 等 | 为什么停止 |
| 语义结果 | answerable / insufficient | 成功判断后，证据是否足够 |
| 评测 | raw_answer / extracted_answer / score | 最终回答、抽取结果和 GT 比较 |

具体规则：

- 到达最大轮数但没有有效最终判断：incomplete，不能自动算成文档不可回答。
- 最后一轮刚获得图片：记录工具返回尚未被后续 Locator 消费。是否预留收尾轮，作为显式预算策略。
- Reasoner 返回 answerable：还应检查 answer 非空、格式正确，并记录判断所用 evidence_version。
- Reasoner 技术失败：进入技术重试 / 错误流程，不能自动成为语义 insufficient。
- pipeline 的兜底补调：保留则标记来源并计入总调用成本。
- 对已有 benchmark 若仍需把技术失败映射为错误分或某个输出字符串，保持统一计分规则，同时单独报告技术失败率，不静默丢弃难题。

现有 [pipeline_response_text][eval-answer] 会把 abstain 和空答案转换为 Not answerable。技术失败可能最终表现成模型拒答，需要在诊断记录中保留原因，避免误读。

## 10. 日志与可回放性

建议每题保存事件日志及最终结果，包含：

- 样本 ID、文档 ID、问题、模型标识、模板来源和生效配置。
- 每次请求的消息序列或可重建该序列的事件引用。
- assistant content、服务实际返回的 reasoning、真实 tool_calls 和原始 arguments。
- 每个 tool_call_id 对应的执行结果、耗时、错误类型和实际返回页面。
- finish_reason、token usage、模型请求耗时及总预算。
- evidence_version、Selector / Reasoner 事件、压缩事件和兜底调用。
- 图片资源路径、哈希、实际发送尺寸、顺序、是否为全页或 crop。
- final_answer、终止原因及单独的评分结果。

普通日志不必重复存储所有 base64，可用受版本管理的图片清单和哈希重建请求；但真正发给模型时仍须包含可读取的图像内容。仅有图片数量不能实现回放。

reasoning 原文可用于内部实验回放，日常统计只需摘要或长度；它不应混入最终答案或答案抽取文本。

## 11. 分文件修改清单

以下均是拟修改项，本次不实施。

| 文件 | 拟修改内容 | 优先级 |
| --- | --- | --- |
| locator.py | 每题初始化一次历史；追加真实 assistant 和逐调用 tool；取消每轮伪用户状态重建；明确终止策略 | P0 |
| llm.py | 返回完整结束信息；增加历史序列化适配；显式 thinking、超时和输出预算配置 | P0 |
| locator.py 中日志函数 | 保留 tool_calls / tool_call_id / reasoning；补齐图片资源引用和请求元数据 | P0 |
| config.py | 新增会话模式和 Locator thinking 等配置；更新“图片必须用 user”旧注释；记录实际生效配置 | P0 |
| retriever.py | 明确页号和 k；严格验证参数；区分 OCR / 图像可用性，支持图像回查 | P1 |
| reasoner.py | 区分执行错误与证据不足；校验 answerable 输出 | P1 |
| pipeline.py | 区分完成、预算耗尽、技术错误；标记兜底与证据版本 | P1 |
| run_mmlongbench.py | 保留原始失败原因、工具完成率、推理答案和抽取结果；保持可对照评分规则 | P1 |
| memory.py / 上下文管理模块 | 分离外部证据与请求视图；后续增加 token 预算与明确压缩策略 | P2 |
| 训练数据生成流程 | 修正输入轨迹表示、保留图像资源引用，验证旧检查点兼容性 | P2 |

建议保留明确的 legacy_state_rebuild 与 native_tool_history 两种实验模式，结果写入不同目录。先以小样本验证，再决定是否将新模式设为默认；不要覆盖旧结果。

## 12. 验证方法与验收标准

### 12.1 不调用模型的结构验证

以下检查应全部通过：

- 每个 assistant.tool_calls 中的调用，都有且仅有一个对应的 tool 返回。
- 一个调用返回多张图，仍对应一个 tool_call_id。
- 同轮多个调用分别返回，调用顺序和图像归属不混淆。
- 单题循环没有凭空生成的额外用户问题；真正的用户追问仍作为新 user 处理。
- 错误参数、工具异常、缺失图片和空检索都有明确返回。
- 保存 / 回放不丢失调用 ID、reasoning 字段及图片资源顺序。
- 0-based 内部页号与 benchmark 1-based 证据页转换正确。
- 本地模板渲染后，标准工具链中的人工 reasoning 标记仍存在。
- 输出截断不会被误判为正常完成。

### 12.2 小规模真实模型验证

按顺序执行：

1. 多组受控图片测试：工具真实返回图片，检查颜色或位置识别；加入互换颜色、不同排列和无图对照，减少猜对的可能。
2. 两次连续 read_page：第二轮回答需要联合两轮图像，确认历史图像可见性。
3. 工具参数错误恢复：模型收到真实 tool 错误后重新调用，不丢失原问题。
4. 跨页 QA：记录读到的证据、最终判断和答案是否正确。
5. 轮数 / 上下文 / 超时测试：确认停止原因和评分映射准确。

受控图片测试只能验证具体部署组合中的能力，不能外推到所有 Qwen3.5 检查点、所有 API 或所有图片难度。

### 12.3 利用已有测试，但区分“链路成功”和“答对”

已有 [tool 图片测试结果][tool-image-result] 显示：

- 模型真实调用 get_test_image；
- role=tool 中返回图片后，正确识别左黄、右白；
- user 图片对照也正确。

这支持当前环境可以处理工具图片，不再有必要因为“tool 不能带图片”的假设而改发 user。

已有 [read_page Agent 测试汇总][agent-summary] 则明确注明：完成循环不代表答案正确。该次 4 题结果为：

| 样本 | 链路结果 | 回答情况 |
| --- | --- | --- |
| [0281](/media/data1/leizh/longdoc/output/read_page_agent/test_20261023_104017_2608057/sample_0281.json) | 8 轮后耗尽预算 | 无最终答案 |
| [0283](/media/data1/leizh/longdoc/output/read_page_agent/test_20261023_104017_2608057/sample_0283.json) | 正常结束 | 回答 4，与 GT 5 不符 |
| [0289](/media/data1/leizh/longdoc/output/read_page_agent/test_20261023_104017_2608057/sample_0289.json) | 正常结束 | 回答数据集数量 2，未回答总 item 数 8980 |
| [0291](/media/data1/leizh/longdoc/output/read_page_agent/test_20261023_104017_2608057/sample_0291.json) | 正常结束 | 回答 990.48，与 GT 466.43 不符 |

这些结果说明：修正工具消息链路是基础工程修复，后续仍需解决视觉识别、问题理解、证据关联及停止策略。不能把这 4 题当作总体准确率估计。

## 13. 如何评估这次修复是否有收益

建议先选固定的 20～50 题，覆盖短文档、图表、跨页和多轮回查；通过后扩展到更大集合。

主对照：

| 组别 | 消息机制 | 目的 |
| --- | --- | --- |
| A | 当前 legacy 状态重建 | 保留原始基线 |
| B | 完整原生工具历史，reasoning 续接，图片不隐式删除 | 测量整体协议与历史修复效果 |
| C | 与 B 相同，但按明确策略去除历史 reasoning | 测量 reasoning 自身的贡献 |
| D | 与 B 相同，加入有记录的图像 / 文本压缩 | 测量预算节省与信息损失 |

必须说明：A → B 同时改变了角色结构、动作历史和可见信息，不能把全部差异归因于 role 字段。单独的 user / tool 问题边界机制可以用本次离线模板实验验证；准确率影响需要进一步控制输入信息量。

固定条件包括：模型及检查点、模板、thinking、图像目录和实际尺寸、工具集合、OCR / crop、最大轮次、输出预算、检索参数、温度 / seed、答案抽取器、评分脚本及题目集合。

同时统计：

- 最终 QA 分数；
- 正常工具链完成率和最终答案率；
- 技术失败率、超时率和轮数耗尽率；
- 重复读页 / 重复检索次数；
- 证据页命中情况，仅用于离线分析；
- token、请求耗时、工具耗时和每题总成本。

GT 和证据标注只能用于离线评分，不能参与 Agent 在线选页或决策。旧训练检查点与基础模型应分开报告，不把检查点差异算作协议修复收益。

## 14. 其他顺手发现的配置问题

这些不是本次消息链路问题的主因，建议另列修复项，避免混在一次准确率对照中。

1. read_page 工具 schema 的页数描述写死“最多 3 页”，实际执行上限来自 READ_PAGE_MAX。若覆盖该配置，模型看到的规则可能与执行不同。应从同一配置生成工具说明。
2. 读页归一化会调用 int，可能把浮点和布尔值当作合法页号，应做严格类型检查。
3. 元素级 select_evidence 分支检查元素是否存在于文档，但没有严格要求它曾被 read_page 返回。该分支与工具说明不一致；页级默认链路应单独看待，不能混为所有模式都有同一问题。
4. config 中 PAGE_LEVEL_EVIDENCE 默认开启，而 GLOBAL_EVIDENCE_SELECTION 默认关闭；实际 _tool_call_reasoner 在页级开启时仍触发选择。建议统一说明和实际生效配置，避免仅根据某个开关推断流程。
5. evidence E 的文本和 crop 总量缺少统一 token 预算；仅调 24000 字符的读页预算不足以控制整个请求长度。

## 15. 建议实施顺序

**第一步：冻结旧基线。** 保存配置、模型 / 模板标识、固定题集和原始结果，允许复跑旧会话模式。

**第二步：只修消息链路。** 追加真实 assistant 与 tool，正确返回图片和调用 ID，透传结束原因，显式配置 thinking，补齐可回放日志。

**第三步：修错误与终止语义。** 区分技术失败、证据不足和预算耗尽，记录框架补调与总成本。

**第四步：小样本对照。** 先确认链路正确，再测答案和效率，不以“模型说出了答案”作为验收标准。

**第五步：再做上下文压缩和训练迁移。** 在正确基线上决定哪些 reasoning、图片和历史可以删除，并验证旧检查点是否需要适配。

本次最关键的改动方向是：**把工具返回恢复为可追溯的观察，把同一道题的行动与观察形成连续历史；保留哪些信息，再通过预算和实验决定。**

---

## 本地证据索引

以下链接对应审计时的文件与行号；后续修改代码后行号可能变化。

[locator-build]: /media/data1/leizh/longdoc/aaai_commit_20260803/aaai_commit/citeagent/locator.py:989
[locator-loop]: /media/data1/leizh/longdoc/aaai_commit_20260803/aaai_commit/citeagent/locator.py:1114
[locator-user-result]: /media/data1/leizh/longdoc/aaai_commit_20260803/aaai_commit/citeagent/locator.py:1270
[locator-nudge]: /media/data1/leizh/longdoc/aaai_commit_20260803/aaai_commit/citeagent/locator.py:1134
[locator-reasoner]: /media/data1/leizh/longdoc/aaai_commit_20260803/aaai_commit/citeagent/locator.py:819
[locator-dump]: /media/data1/leizh/longdoc/aaai_commit_20260803/aaai_commit/citeagent/locator.py:273
[llm-tools]: /media/data1/leizh/longdoc/aaai_commit_20260803/aaai_commit/citeagent/llm.py:118
[config-context]: /media/data1/leizh/longdoc/aaai_commit_20260803/aaai_commit/citeagent/config.py:55
[config-pages]: /media/data1/leizh/longdoc/aaai_commit_20260803/aaai_commit/citeagent/config.py:134
[reasoner-entry]: /media/data1/leizh/longdoc/aaai_commit_20260803/aaai_commit/citeagent/reasoner.py:84
[reasoner-errors]: /media/data1/leizh/longdoc/aaai_commit_20260803/aaai_commit/citeagent/reasoner.py:105
[pipeline-decision]: /media/data1/leizh/longdoc/aaai_commit_20260803/aaai_commit/citeagent/pipeline.py:126
[pipeline-fallback]: /media/data1/leizh/longdoc/aaai_commit_20260803/aaai_commit/citeagent/pipeline.py:84
[eval-answer]: /media/data1/leizh/longdoc/aaai_commit_20260803/aaai_commit/citeagent/run_mmlongbench.py:56
[eval-trace]: /media/data1/leizh/longdoc/aaai_commit_20260803/aaai_commit/citeagent/run_mmlongbench.py:66
[qwen-query]: /media/data1/leizh/models/Qwen3.5-9B/chat_template.jinja:67
[qwen-reasoning]: /media/data1/leizh/models/Qwen3.5-9B/chat_template.jinja:89
[qwen-tool]: /media/data1/leizh/models/Qwen3.5-9B/chat_template.jinja:131
[vllm-protocol]: /media/data1/leizh/anaconda3/envs/qwen35/lib/python3.12/site-packages/vllm/entrypoints/openai/chat_completion/protocol.py:470
[vllm-chat]: /media/data1/leizh/anaconda3/envs/qwen35/lib/python3.12/site-packages/vllm/entrypoints/chat_utils.py:1752
[train-locator]: /media/data1/leizh/longdoc/aaai_commit_20260803/aaai_commit/training_data/locator_train.jsonl:1
[simple-agent]: /media/data1/leizh/longdoc/test_qwen35_read_page_agent.py:73
[tool-image-result]: /media/data1/leizh/longdoc/output/tool_image_test_20261022_215444_793273/result.json
[agent-summary]: /media/data1/leizh/longdoc/output/read_page_agent/test_20261023_104017_2608057/summary.json
