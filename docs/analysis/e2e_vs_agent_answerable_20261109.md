# 可回答题：e2e 与 Agent 的逐题对比及框架问题分析

分析日期：2026-11-09。本报告只分析标准答案不是 `Not answerable` 的 **859 道题**；其余223道题不参与任何下列正确率或分组统计。原始实验结果未被改写，也未重新调用模型。

## 1. 对照实验与“答对”的定义

| 项目 | e2e | Agent |
| --- | --- | --- |
| 结果目录 | `output/mmlongbench/qwen35_a100_doc_prefix_mm4_rolling_seed0` | `output/agent/a100_4gpu_direct_pages` |
| 模型记录 | `/media/data5/leizh/models/Qwen3.5-9B` | 同一路径的Qwen3.5-9B |
| 输入 | 全文整页图；超长时按整页保留前缀 | Locator取证，Reasoner接收选定整页图、OCR及可用crop |
| 上下文上限 | 262144 | 65536 |
| thinking | 回答和抽取均关闭 | Locator开启；Reasoner和抽取关闭 |
| temperature / seed | 0 / 0 | 各角色0 / 42 |
| 单次回答生成上限 | 8192 | Locator和Reasoner均8192 |
| 抽取 | 同模型，最多2048 tokens | 同模型，最多2048 tokens |
| Agent轮数预算 | 无此循环 | 最多15轮Locator调用 |

按 `_sample_index` 对齐，并逐项校验 `doc_id/question/answer/answer_format`。859道可回答题全部一致。使用Agent冻结题集中的标准证据页，避免把Agent选页误当成标准页。目录名含a100，但实际显卡类型不作为本报告的能力归因依据。

本报告区分三个评分档：**满分=score为1；部分分=0<score<1；零分=score为0**。官方字符串指标ANLS允许部分分，因此“平均官方得分”和“满分率”不同。先集中分析“满分↔零分”的强分歧，再列出涉及部分分的其他分歧，避免把0.99说成完全不会答。

这不是严格控制变量的架构消融：输入信息、提示词、thinking、上下文和请求批次存在差异，且e2e没有Agent同样的完整代码/权重快照。下文把日志可直接证明的失败点与尚需消融确认的原因分开。

## 2. 总体关系

| 仅859道可回答题 | e2e | Agent |
| --- | ---: | ---: |
| 平均官方得分 | **50.71%** | **42.64%** |
| 满分题数 | 397 | 325 |
| 满分率 | 46.22% | 37.83% |
| 部分分题数 | 48 | 50 |
| 零分题数 | 414 | 484 |

可回答题的平均得分差是 **8.08个百分点**。不能用包括不可回答题的总体Acc差替代这个差距。

| e2e结果 \ Agent结果 | 满分 | 部分分 | 零分 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 满分 | 257 | 11 | **129** | 397 |
| 部分分 | 4 | 29 | 15 | 48 |
| 零分 | **64** | 10 | 340 | 414 |
| 合计 | 325 | 50 | 484 | 859 |

两边有互补性：如果离线用标准答案逐题取两者得分的较大值，平均得分为59.38%，有465题至少一方满分。这只是使用真值的互补性上界，不是已实现的路由器成绩。

在129道“e2e满分、Agent零分”中，Agent的停止原因如下：

| 停止原因 | 题数 | 直接含义 |
| --- | ---: | --- |
| 正常回答并完成抽取 | **95** | 需要继续区分答错、抽错和评分格式 |
| 达到15轮上限 | **20** | 未产出最终答案；其中13题根本没有发出Reasoner请求 |
| 输出截断 | **5** | Locator thinking耗尽8192 tokens |
| Reasoner答案字段类型被拒绝 | **9** | 返回数组，而接口只接受字符串 |

因此，增大轮数或增加截断重试只能处理其中一部分；95道正常完成但零分的案例同样重要。全部859道中，Agent有86道触顶、16道截断、12道类型失败，其余745道正常结束。

按答案类型统计：

| 类型 | 题数 | e2e平均分 | Agent平均分 |
| --- | ---: | ---: | ---: |
| Int | 299 | 42.14% | 34.78% |
| Float | 159 | 60.38% | 51.57% |
| Str | 250 | 59.46% | 55.84% |
| List | 151 | **43.02%** | **26.91%** |

列表题下降最明显，与类型拒绝、答案列表化失败、漏项/多项及集合比较错误同时出现，不能全部归因为视觉定位。

## 3. 从日志中确认的框架问题

### 3.1 答案正确，抽取却把它改错

- **#44**：Reasoner原文已经是正确的 `Analysis`，抽取结果却是 `Not answerable`。
- **#511**：原文 `Start new search`，抽取结果同样变成 `Not answerable`。
- **#5 / #498 / #564**：问题问群体或项目名称，原文包含正确名称，抽取器却只保留关联数字 `8% / 795 / 27`。
- **#93 / #965**：正确的多个实体被保留成一个普通句子，官方List评分将其视为单元素，长度不一致而零分。
- **#0**：Agent语义上答到less well off，但抽取整句话导致零分；e2e为0.9231部分分，此题不计入129道强分歧。

两种方法抽取器的模型、提示词文件内容和生成额度相同，提示词SHA256同为 `96a22c3003c22801f1f19717be45070b1d6b9ce76b42e171e044f63b641de659`，**但请求组织方式不同**：

```text
e2e:  user = 官方抽取提示词
      assistant = Question: ...\nAnalysis: ...

Agent: system = 官方抽取提示词
       user = Extract from the following data: {question, model_response}
```

这使当前对比混入了抽取协议差异。它也并非只伤害Agent：**#58的e2e原文就是正确的Appota，却被其抽取器改为Not answerable**。不应把这类分差当成某种取证架构的优势。

改进：统一两边的抽取请求协议、输出类型和单位约定；先保存并固定现有raw answer，只重跑抽取作为单独实验。保留raw/extracted/normalized三层结果，检查单词答案被改成弃答、名称题抽成数值、列表题变单字符串等行为。抽取器仍只接收问题与模型输出，不接收标准答案或标准证据页。

### 3.2 数值格式被当成内容错误

**#1016**：标准1358000，Agent原文和pred为`1,358,000`；**#1031**：标准1375500，pred为`1,375,500`；**#525**：问题的Int标准值32，pred为`32%`。这些值与任务所需数字相容，却不能被原版数值解析器直接解析，得到零分。

改进：在抽取输出层明确生成无千分位的规范数值，按问题要求保留单位；不要修改官方评分公式来掩盖差异。可以对规范的千分位数字做可审计的无损规范化，但不能一概删除单位或百分号，因为#418的`399 million→399`恰恰丢失了关键信息。两种方法使用同一套规范。

### 3.3 合法JSON数组被接口拒绝

129道强损失中9题触发这一错误。**#562**原始返回`{"status":"answerable","answer":["46","27","64"]}`，列表内容与标准一致；程序要求answer必须是str，直接终止。**#675**也返回了正确城市列表，但未进入抽取。另一些数组内容本来就不正确，所以不能把全部9题直接补成1分。

改进：把合法答案列表无损序列化成文本，或正式扩展答案类型；其他字段错误允许一次结构纠正。当前JSON重试只覆盖解码失败，数组字段是合法JSON，后置类型校验不走该重试。规范化不得从标准答案补项、排序或修改内容。

### 3.4 页号的输入和输出存在混用

- **#409**：问题指定slides 12和14，Agent直接调用`read_page([12,14])`；标准对应的物理0-based页是11和13，读成了后两页，答案只有Interphase和Cytokinesis。
- **#581 / #236**：模型返回内部页索引0/8，题目和标准需要人类页号1/9。
- **#540**是成功修正的反例：最先读物理113，随后通过检索定位Figure 113实际在物理142，而不是把图号当页号。

改进：工具观察同时提供物理0-based索引、物理1-based页码、可识别的印刷页号；对问题中的“page/slide/Figure”先解析引用类型，再调用read_page。输出页码也走明确约定。映射来自PDF/OCR，标准证据页只用于离线诊断，不能喂给Locator。

### 3.5 读到了证据，仍迟迟不调用Reasoner

**#36**：read_pages已包含全部标注页，15轮用在6次search和9次read_page上，Reasoner请求为0；**#119**类似。**#28 / #76**已读到关键图页，仍在thinking中循环到8192 tokens截断。

129道强损失中，按标准页归一化后，有11题已经读全标注页但未发Reasoner请求。标注页空缺/无效的4题未参与这个覆盖判断。

改进：为提交证据预留轮次；把总轮数、剩余预算和取证进展明确传给Locator；在末轮可约束其选择call_reasoner，但选页仍由Locator完成并记录为控制器干预。对相同search/read的无进展重复做诊断。增加thinking预算控制及一次受控截断恢复。改变这些策略后使用新实验目录，不把旧零分样本静默改成成功。

### 3.6 已读证据在提交时被遗漏，或选了错误的表/指标

**#183**是清晰的提交失误：Locator读过物理15和18，两页分别给出Math=45.0、Computer Science=57.1；但三次Reasoner请求只包含`[16,19]`、`[16,19]`、`[5,16,19]`。最终回答Art & Design与Science之差，题目实体已经变了。

同时，**不能只凭gold page覆盖就定性为检索错误**：**#761**虽然没有提交标注物理6，但物理5已经有正确的Table 1（已人工看图），并且实际交给Reasoner。Reasoner仍引用另一张Table 7，回答6种而不是4种。这主要是表格选择/比较口径问题，且标准页本身可能存在偏移。

改进：维护“待解决事实—候选页—已提交页”的显式记录；提交前核对问题实体、年份、指标和此前missing。不要不加选择地累加所有读过的页，也不要将替换证据的工具行为本身当成bug。现协议每次call_reasoner替换页集合是设计行为，遗漏必须通过证据管理解决。

### 3.7 算术、集合比较和答案内部一致性缺少校验

- **#924**：读出43和22，却写“下降30”；应为21。
- **#1080**：写出资产127+19、负债22+31，却把后者算成38；应为146−53=93。
- **#964**：两组生命周期阶段已列对，但集合差得出Concept和Inception；应为Concept和Production。
- **#473**：列出11个菜单功能，句首却说10个。
- **#642**：自己列出16、17、33篇论文，却把最多的类别答为16篇对应的类别。
- **#529**：把自己已经列为LinkedIn包含项的Post Types又判为“不包含”。

改进：优先增加有限的计算、集合差/交、去重计数工具或结构化中间结果校验；每个操作数附来源页，再执行运算。仅靠“更认真地推理”的提示不能可靠修复。验证不使用标准答案，并保留校验失败和纠正成本。

### 3.8 同一份证据上的insufficient反复循环

**#1034**：六次Reasoner请求都使用`[58,60]`，多次要求“找到payables turnover的公式或值”，直到预算耗尽。e2e则从财务原始数据计算出9.67。当前缺少对“相同证据＋相同missing”的进展控制，以及对“可由已知原始量推导”的明确处理。

改进：区分缺少操作数、缺少可用公式、视觉辨识不清和不可计算；同证据重复insufficient时要求具体缺哪一行/列，不再笼统搜指标名称。允许有依据的算术推导，不能把“文档没有直接写出派生指标”一律当证据不足。具体该比率所需数据是否已在这两页齐备，仍需逐字段复核，不能只凭gold页断言。

## 4. Agent为什么有些题更好

以下是与现有日志一致的解释，不把它们当作已完成的消融结论。

1. **局部视觉细节与OCR/crop辅助**：#540在166页文档中定位到物理142，仅交一页整图和一个crop，认出YouTube Music；人工看图确认图标文字很小。#106只交目标餐桌页及7个crop，数出7个酒杯，e2e为6。只有关掉crop/OCR的对照才能量化各自贡献。
2. **选对目标图，而非被全文相似内容带偏**：#803把ValueEdge的Plan/Build/Test/Deliver/Run图交给Reasoner；e2e答成另一页的五条平台价值陈述。#103补齐e2e漏掉的College of Engineering。#869补齐漏掉的Reasoning Error类别。
3. **将跨页操作数集中后计算**：#607提交物理36/37，图上Hospitality一行分别为61%与61%，得到122；已人工看图。e2e这题陷入编号重复并截断。#1059把现金流和债务页集中，计算1469502/1907231≈0.77；e2e甚至找到了相关原始量，却因没有直接给出比率而拒答。
4. **部分“优势”来自抽取或评测表达**：#58、#174、#418、#531、#593、#777、#988共7个已核对实例，e2e已答到目标内容，但最后抽取/表达不合评分预期。不能把这7题算作纯取证优势；完整案例见下文。

64道Agent强优势题全部都曾向Reasoner发出请求；59题至少一次请求包含全部可用标准证据页，另5题也不等于“没证据”，因为标准页可能不完整或存在替代证据。e2e这64题中有21题最终预测Not answerable，原因既包括实际漏看/误读，也包括抽取错误和问题解释过于保守。

## 5. 不能直接当成优质成功轨迹的Agent满分题

以下6个案例是已发现的解释一致性问题，不声称这是全部不一致样本，也不改动其官方分数：

| 编号 | 已保存的满分预测 | 原始答案中的问题 |
| --- | --- | --- |
| 12 | 7 | 分项为1+4+3+1+2=11，却总称7；并声称没有表格 |
| 345 | 16 | 首句说13，后面45−29又说16 |
| 568 | 15 | 女性所列69+38+24+5+5+8应为149，男性61+53+30+5+5+10应为164；却写成114与99 |
| 621 | 3 | 将`Bryce Harper 1.wmv/2.wmv`两个文件名当作website URLs计数 |
| 832 | 10 | 首句说7，接着枚举了10个学习成果 |
| 927 | 29 | 首句说19，后面48−19又得29 |

这些例子说明：最终短答案得分正确不保证过程可复用。若今后用轨迹做SFT/GRPO，除结果奖励外，应检查运算和证据一致性；不能把有矛盾的解释直接当成高质量推理标注。

## 6. 修改顺序与验证方式

| 优先级 | 改动 | 本次支持证据 | 验证要求 |
| --- | --- | --- | --- |
| P0 | 统一抽取协议、字段和数值表示 | #44/#511/#58/#1016/#525；两边同prompt不同messages | 固定raw answer，仅重抽取；分别报告原始与新分数 |
| P0 | 合法列表无损规范化；类型纠正重试 | 9道强损失；#562/#675 | 不使用GT，重走原抽取；不保证9题全恢复 |
| P0 | 明确物理/印刷/用户页号 | #409/#581/#236/#540 | 人工核对映射，显式指定页和返回页号题回归 |
| P1 | thinking预算、一次受控恢复、提交轮次预留 | 5道截断、20道触顶强损失 | 相同题集检查原失败和正常题，报告额外成本 |
| P1 | 显式证据清单与missing进展检查 | #183/#818/#1034 | 记录各次提交页，关注事实覆盖而非只看gold命中 |
| P1 | 算术、集合、计数一致性校验 | #924/#1080/#964/#473/#642 | 操作数有出处；不只检查最终字符串格式 |
| P1 | 保存gold与agent证据为不同字段 | merged.evidence_pages被Agent覆盖 | `gold_evidence_pages`只供离线报表；不影响推理输入 |
| P2 | 增强小字/图表读数与定向crop | #540成功、#547/#588失败 | 同图同问题对照，避免在已有定位/格式bug上盲目换模型 |

不建议首先只增加到30轮：部分样本在已有证据上反复搜索或重复insufficient，会扩大耗时。先修接口和抽取，再做预算实验，才能判断Agent本身有多少空间。

建议保持现有859题固定，逐次开展：①固定raw answer统一抽取；②在其基础上加类型和页号修复；③加预算与进展控制；④加计算校验；⑤独立对照只保留整图、整图+OCR、整图+OCR+crop。每个版本新目录，既回归236个得分分歧样本，也检查257个共同满分样本是否退化。正式全量仍用同一官方评分，不按分析时已知GT选择e2e或Agent输出。

## 7. 审计边界与数据注意事项

- 本报告读取了全部236个得分不同样本的1439条工具事件和实际Reasoner请求摘要。重点案例额外核对原始回答、抽取结果与部分页图；附录的自动诊断标签不是逐题人工视觉定论。
- 129道强损失中有73题至少一次实际Reasoner请求覆盖全部可用标注页，21题未读到任何标注页，11题读全但未请求Reasoner，3题读全却未一起提交。以上是可重叠的诊断计数，不应相加成129；4题标注为空或包含0，排除页覆盖推断。
- 标准证据页并非绝对真值：#761的正确表实际位于物理5，而标注换算后为6；#927也存在标注页与实际数据页不一致的信号。页未命中不等于一定无证据，页全命中不等于视觉读数正确。
- e2e有4道可回答题发生整页前缀截断：#557/#558/#560/#561，468页只保留310页。这4题不在193道满分↔零分强分歧中，因此不能用输入截断解释上述强分歧。
- 以另一完整e2e目录`qwen35_a100_doc_prefix_mm4`做敏感性检查，可回答题平均分50.46%，强分歧为128道e2e优势、63道Agent优势，主要结论相近；下文具体名单始终以rolling_seed0为准。
- 所有页号集合除特别注明外均为物理0-based；问题中的人类页码与图号保留原文。标准答案只用于本次离线评价与审计。

下方给出重点案例与完整分歧清单。完整逐题原文和元数据另保存在工作区`output/analysis/e2e_vs_agent_answerable_20261109/`，不复制图片。

<!-- GENERATED APPENDICES -->

## 8. 重点案例原文对照

### #0：部分分补充：语义近似正确，但长答案抽取归零

问题：According to the report, how do 5% of the Latinos see economic upward mobility for their children?

- 标准答案：`Less well-off`
- e2e：score=0.9231，pred=`"Less well off"`
- Agent：score=0.0000，pred=`5% of Latinos expect their children to be less well off financially than they themselves are now.`；stop=`reasoner_answerable`，4轮。
- 标准证据页（物理0-based）：`[4]`；已读：`[4, 16]`；最终选页：`[4, 16]`。

e2e原文：

> The pie chart on page 5, titled "Latinos see economic upward mobility for their children," illustrates the percentage of Latinos who believe their children will be financially better off, about the same, or less well off. The chart shows that 72% expect their children to be "Better off," 16% expect them to be "About the same," and 5% expect them to be "Less well off." Therefore, 5% of Latinos see their children as being less well off financially.

Agent原文：

> According to the report, 5% of Latinos expect their children to be less well off financially than they themselves are now.

实际Reasoner请求摘要：

```json
[{"pages": [4, 16], "full_images": 2, "crops": 2, "ocr_characters": 4855, "status": "answerable"}]
```

分析：e2e得0.9231而非满分，此题不计入129。Agent回答到less well off，抽取却保留整句。ANLS测字符串距离，不是对原推理的语义判定。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0000/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0000/attempt_0001.jsonl)

### #5：名称题被抽成百分比

问题：According to this report, among the Hispanics shaped by four current personal financial situation, which situation involves the highest percentage that expects their future financial situation to get a lot worse?

- 标准答案：`Poor Financial Condition`
- e2e：score=1.0000，pred=`Poor financial condition`
- Agent：score=0.0000，pred=`8%`；stop=`reasoner_answerable`，3轮。
- 标准证据页（物理0-based）：`[15]`；已读：`[10, 12, 15]`；最终选页：`[15]`。

e2e原文（节选）：

> Page 16 contains a bar chart titled "Future financial expectations among Hispanics shaped by current personal financial situation". This chart shows the percentage of Hispanics who expect their family's finances to improve or worsen, broken down by their current financial condition. The legend indicates that the lightest brown bar represents "Get a lot worse". Looking at the bars for each category, the "Poor financial condition" group has the longest bar for "Get a lot worse", with a value of 8%. The other categories have lower percentages for this outcome (Excellent: 5%, Good: 1%, Only fair: 3%). Therefore, those with a poor financial condition are most likely to expect their situation to g…

Agent原文：

> According to the chart, among Hispanics with a poor financial condition, 8% expect their future financial situation to get a lot worse, which is the highest percentage among the four current personal financial situations (Excellent: 5%, Good: 1%, Only fair: 3%, Poor: 8%).

实际Reasoner请求摘要：

```json
[{"pages": [15], "full_images": 1, "crops": 1, "ocr_characters": 2043, "status": "answerable"}]
```

分析：两边原文都明确poor financial condition且都提到8%，区别发生在最终抽取。应修目标类型约束，不能要求Locator继续搜索。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0005/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0005/attempt_0001.jsonl)

### #44：单词正确答案在抽取阶段被改成弃答

问题：According to the flowchart "Analytics Value Chain", what comes between Reporting and Action?

- 标准答案：`Analysis`
- e2e：score=1.0000，pred=`Analysis`
- Agent：score=0.0000，pred=`Not answerable`；stop=`reasoner_answerable`，3轮。
- 标准证据页（物理0-based）：`[12]`；已读：`[12]`；最终选页：`[12]`。

e2e原文：

> Based on the "Analytics Value Chain" flowchart shown in Page 13, the stage that comes between "Reporting" and "Action" is "Analysis". The flow is depicted as Data -> Reporting -> Analysis -> Action -> Value.

Agent原文：

> Analysis

实际Reasoner请求摘要：

```json
[{"pages": [12], "full_images": 1, "crops": 1, "ocr_characters": 407, "status": "answerable"}]
```

分析：原文恰为Analysis；Reasoner已完成任务，抽取器引入错误。原始调用见trace中Reasoner和extractor响应，不存在从标准答案补写。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0044/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0044/attempt_0001.jsonl)

### #93：正确两个实体未变成列表

问题：What devices other than phone are introduced for setting device+?

- 标准答案：`['vision', 'bluetooth device']`
- e2e：score=1.0000，pred=`['Vision', 'Bluetooth device']`
- Agent：score=0.0000，pred=`Vision and Bluetooth device`；stop=`reasoner_answerable`，5轮。
- 标准证据页（物理0-based）：`[10, 11]`；已读：`[10, 11, 22, 23, 24]`；最终选页：`[10, 11, 22, 23, 24]`。

e2e原文：

> The document provides a list of device types that can be linked with a phone using the Device+ feature. On page 11 (image 11), under the section "Set Device+", it states that the feature supports linking phones with "Vision" devices. Page 12 (image 12) continues this list, mentioning "Bluetooth device" as another type of device that can be linked. Therefore, the other device types are Vision and Bluetooth devices.

Agent原文：

> Vision and Bluetooth device

实际Reasoner请求摘要：

```json
[{"pages": [10, 11, 22, 23, 24], "full_images": 5, "crops": 3, "ocr_characters": 13303, "status": "answerable"}]
```

分析：Vision和Bluetooth device都在原文，但pred仍为普通字符串。官方List评测按一个元素处理，数组长度不符而零分。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0093/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0093/attempt_0001.jsonl)

### #1016：正确股份数量被千分位格式挡住

问题：How much shares did COSTCO repurchase in FY2021?

- 标准答案：`1358000`
- e2e：score=1.0000，pred=`1358000`
- Agent：score=0.0000，pred=`1,358,000`；stop=`reasoner_answerable`，3轮。
- 标准证据页（物理0-based）：`[28]`；已读：`[28, 54]`；最终选页：`[28, 54]`。

e2e原文：

> The answer is found in the 'Stock Repurchase Programs' section on page 29. The text states, 'During 2021 and 2020, we repurchased 1,358,000 and 643,000 shares of common stock...'. This indicates that 1,358,000 shares were repurchased in FY2021. This information is also presented in a table on page 55 under 'Note 7—Equity', which shows '1,358' in the 'Shares Repurchased (000's)' column for the year 2021.

Agent原文：

> COSTCO repurchased 1,358,000 shares in FY2021, as stated in the 'Stock Repurchase Programs' section on page 29 and confirmed in the table on page 54 under the 'Shares Repurchased (000's)' column for 2021.

实际Reasoner请求摘要：

```json
[{"pages": [28, 54], "full_images": 2, "crops": 1, "ocr_characters": 8116, "status": "answerable"}]
```

分析：本题answer_format为Float。pred的逗号让float转换失败。抽取层应输出1358000，保留含逗号的原文用于追溯。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1016/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1016/attempt_0001.jsonl)

### #1031：正确财务数值的千分位格式问题

问题：what profit  did Memberships contribute to in Domestic Streaming Segment in FY2015? Answer in thousands.

- 标准答案：`1375500.00`
- e2e：score=1.0000，pred=`1375500`
- Agent：score=0.0000，pred=`1,375,500`；stop=`reasoner_answerable`，3轮。
- 标准证据页（物理0-based）：`[20]`；已读：`[20, 60]`；最终选页：`[20, 60]`。

e2e原文：

> Based on the 'Domestic Streaming Segment' table on page 21, for the year ended December 31, 2015, the 'Contribution profit' is listed as 1,375,500. The table header indicates that the values are 'in thousands, except revenue per membership and percentages'. Therefore, the contribution profit was $1,375,500 thousand.

Agent原文：

> 1,375,500

实际Reasoner请求摘要：

```json
[{"pages": [20, 60], "full_images": 2, "crops": 5, "ocr_characters": 9506, "status": "answerable"}]
```

分析：同样是Float，1375500和1,375,500的数值相同。直接增加推理预算不会改变解析器行为。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1031/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1031/attempt_0001.jsonl)

### #525：百分号与整数输出规范不一致

问题：In 2013, what is the total percentage of online adult users in the "White, Non-Hispanic" and "Hispanic" categories on the platform whose metrics include Followers, Tweet Clicks, Video Views, Tweet Reach, Engagement Rate, Completion Rate, Engagement: Mentions, Retweets, Likes, Top Tweet, Top Mention, Top Follower, Top Media Tweet, and Twitter Referral Traffic?

- 标准答案：`32`
- e2e：score=1.0000，pred=`32`
- Agent：score=0.0000，pred=`32%`；stop=`reasoner_answerable`，3轮。
- 标准证据页（物理0-based）：`[14, 17]`；已读：`[14, 17]`；最终选页：`[14, 17]`。

e2e原文（节选）：

> To answer this question, we need to identify the platform first and then find the relevant data from the document. 1. **Identify the Platform:** The question describes a platform with specific metrics. Let's look at the "Twitter Metrics" slide (Page 18). The metrics listed are: Followers, Tweet Reach, Engagement (Mentions, Retweets, Likes), Tweet Clicks, Engagement Rate, Top Tweet, Top Mention, Top Follower, Top Media Tweet, Video Views, Completion Rate, and Twitter Referral Traffic. This list matches the one provided in the question. Therefore, the platform is Twitter. 2. **Find the Data:** Now we need to find the percentage of online adult users for "White, Non-Hispanic" and "Hispanic" cat…

Agent原文：

> 32%

实际Reasoner请求摘要：

```json
[{"pages": [14, 17], "full_images": 2, "crops": 1, "ocr_characters": 2735, "status": "answerable"}]
```

分析：正确32%被原样交给Int解析，导致零分。只对问题所需输出类型做表示规范，不能普遍抹掉单位。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0525/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0525/attempt_0001.jsonl)

### #562：正确JSON数组在答案类型校验处退出

问题：In 2015, how many percentage of surveyed adults, Republicans, and Democrats believed that the goverment was doing very/fairly well in reducing the threat of terrorism? Please write the answer in list format, e.g., ["3","2"]

- 标准答案：`['46', '27', '64']`
- e2e：score=1.0000，pred=`[46, 27, 64]`
- Agent：score=0.0000，pred=`—`；stop=`invalid_reasoner_answer`，3轮。
- 标准证据页（物理0-based）：`[1, 10]`；已读：`[10, 11]`；最终选页：`[10, 11]`。

e2e原文：

> Based on the table titled "Older, less educated give more negative ratings of govt efforts against terrorism" on page 11 (image 11), the percentages of surveyed adults who believe the government is doing very/fairly well in reducing the threat of terrorism are as follows: - Total (all adults): 46% - Republican: 27% - Democrat: 64% Therefore, the list of percentages for adults, Republicans, and Democrats is [46, 27, 64].

Agent原文：

> —

实际Reasoner请求摘要：

```json
[{"pages": [10, 11], "full_images": 2, "crops": 2, "ocr_characters": 4778, "status": "answerable"}]
```

被拒绝的实际返回：

```json
{"status": "answerable", "answer": ["46", "27", "64"]}
```

分析：实际数组46/27/64与标准内容一致。json.loads已成功，随后非str校验抛出不可重试错误。能修的是接口，不应将所有数组案例无条件判对。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0562/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0562/attempt_0001.jsonl)

### #675：正确城市列表也被接口拒绝

问题：Which cities does the China's largest cloud provider have teams? Write the answer in list format with alphabetical rder.

- 标准答案：`['Dubai', 'Frankfurt', 'Hong Kong', 'London', 'New York', 'Paris', 'San Mateo', 'Seoul', 'Singapore', 'Sydney', 'Tokyo']`
- e2e：score=1.0000，pred=`['Dubai', 'Frankfurt', 'Hong Kong', 'London', 'New York', 'Paris', 'San Mateo', 'Seoul', 'Singapore', 'Sydney', 'Tokyo']`
- Agent：score=0.0000，pred=`—`；stop=`invalid_reasoner_answer`，3轮。
- 标准证据页（物理0-based）：`[4, 5]`；已读：`[5, 14]`；最终选页：`[5]`。

e2e原文：

> Based on the information provided in the document, the China's largest cloud provider, Alibaba Cloud, has teams stationed in the following cities: * Dubai * Frankfurt * Hong Kong * London * New York * Paris * San Mateo * Seoul * Singapore * Sydney * Tokyo

Agent原文：

> —

实际Reasoner请求摘要：

```json
[{"pages": [5], "full_images": 1, "crops": 3, "ocr_characters": 2816, "status": "answerable"}]
```

被拒绝的实际返回：

```json
{"status": "answerable", "answer": ["Dubai", "Frankfurt", "Hong Kong", "London", "New York", "Paris", "San Mateo", "Seoul", "Singapore", "Sydney", "Tokyo"]}
```

分析：Reasoner已给出11个城市的数组，实际答案保存在llm_response中，结果raw_answer却为null。说明审计必须看响应日志，不能只看最终result。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0675/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0675/attempt_0001.jsonl)

### #36：全局排查已经找到证据，却没有提交

问题：How many human quotes are provided with source in this document?

- 标准答案：`5`
- e2e：score=1.0000，pred=`5`
- Agent：score=0.0000，pred=`—`；stop=`max_rounds`，15轮。
- 标准证据页（物理0-based）：`[13, 18, 19, 32, 36]`；已读：`[0, 1, 13, 18, 19, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]`；最终选页：`[]`。

e2e原文：

> Based on the provided document, there are 5 human quotes that are attributed to a specific person or source. Here is a list of each quote and its source: - Page 14: A quote from John Holloway of the European Investment Fund. - Page 19: A quote from Egbert Freiherr von Cramm, Managing Director at Deutsche Bank Private Equity. - Page 20: A quote from Uli Fricke, who was the EVCA Chairwoman from 2010-2011. - Page 34: A quote from Sven Weber, President of Silicon Valley Bank Capital. - Page 37: A quote from Andreas Ritter of ARICO Private Investments.

Agent原文：

> —

实际Reasoner请求数：**0**。

分析：15轮产生6次search和9次read_page，包含全部标准页，但Reasoner请求为0。这是取证与收尾策略的问题，不能仅以“15轮太少”解释。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0036/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0036/attempt_0001.jsonl)

### #28：正确页已读到，thinking继续重复至截断

问题：What is the web address in the orange box in page 47 that gives the Wiki Query Service code?

- 标准答案：`https://w.wiki/6gb`
- e2e：score=1.0000，pred=`https://w.wiki/6gb`
- Agent：score=0.0000，pred=`—`；stop=`output_truncated`，11轮。
- 标准证据页（物理0-based）：`[46]`；已读：`[46, 47, 48]`；最终选页：`[]`。

e2e原文：

> The image on page 47 displays a map of Dutch municipalities. In the bottom right corner, there is an orange box containing a URL. This URL is 'https://w.wiki/6gb'. The final answer is: https://w.wiki/6gb

Agent原文：

> —

实际Reasoner请求数：**0**。

分析：已读46/47/48，金标换算后为46；生成到8192 tokens时没有content与tool_calls。可在保留原记录的前提下测试thinking预算和受控恢复。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0028/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0028/attempt_0001.jsonl)

### #409：用户页码直接当内部索引

问题：Which stages of cell division are shown on slides 12 and 14?

- 标准答案：`['Interphase', 'Prophase', 'Prometaphase', 'Metaphase', 'Anaphase', 'Telophase and cytokinesis']`
- e2e：score=1.0000，pred=`['Interphase', 'Prophase', 'Prometaphase', 'Metaphase', 'Anaphase', 'Telophase and Cytokinesis']`
- Agent：score=0.0000，pred=`["Interphase", "Cytokinesis"]`；stop=`reasoner_answerable`，2轮。
- 标准证据页（物理0-based）：`[11, 13]`；已读：`[12, 14]`；最终选页：`[12, 14]`。

e2e原文：

> Slide 12 displays diagrams for the stages of cell division labeled as INTERPHASE, PROPHASE, and PROMETAPHASE. Slide 14 continues this sequence, showing diagrams for METAPHASE, ANAPHASE, and TELOPHASE AND CYTOKINESIS. Combining the information from both slides provides a complete list of the depicted stages.

Agent原文：

> Slide 12 shows Interphase. Slide 14 shows Cytokinesis (specifically, cleavage in animal cells and cytokinesis in plant cells).

实际Reasoner请求摘要：

```json
[{"pages": [12, 14], "full_images": 2, "crops": 6, "ocr_characters": 1983, "status": "answerable"}]
```

分析：实际read_page参数就是[12,14]，而标准对应物理11/13。这里有明确调用参数证据；不能把所有其他题的gold页偏移都机械作同类判断。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0409/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0409/attempt_0001.jsonl)

### #581：内部页号原样作为用户答案

问题：What page has an eight-pointed snowflake image? Give me an integer.

- 标准答案：`1`
- e2e：score=1.0000，pred=`1`
- Agent：score=0.0000，pred=`0`；stop=`reasoner_answerable`，5轮。
- 标准证据页（物理0-based）：`[0]`；已读：`[0]`；最终选页：`[0]`。

e2e原文：

> 1

Agent原文：

> 0

实际Reasoner请求摘要：

```json
[{"pages": [0], "full_images": 1, "crops": 1, "ocr_characters": 1303, "status": "answerable"}]
```

分析：snowflake位于物理0，题目标准页号1。内部返回page_index=0，对外作答缺少1-based转换。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0581/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0581/attempt_0001.jsonl)

### #183：已经读过正确两张表，提交的却是其他页

问题：According to this paper,  What's the GPT-4V's performance difference between CS subject and math subject on MMMU datasets?

- 标准答案：`12.1%`
- e2e：score=1.0000，pred=`12.1`
- Agent：score=0.0000，pred=`Not answerable`；stop=`reasoner_answerable`，12轮。
- 标准证据页（物理0-based）：`[15, 18]`；已读：`[5, 6, 13, 14, 15, 16, 17, 18, 19]`；最终选页：`[5, 16, 19]`。

e2e原文（节选）：

> To determine the performance difference, we need to find the scores for GPT-4V in both Computer Science (CS) and Math subjects from the provided tables. 1. **Find CS Score:** Table 9 on page 19 shows the 'Tech & Engineering' results. The 'Computer Science' subfield score for GPT-4V(ision) (Playground) is 57.1. 2. **Find Math Score:** Table 6 on page 16 shows the 'Science' results, which includes the 'Math' subfield. The score for GPT-4V(ision) (Playground) in Math is 45.0. 3. **Calculate Difference:** Subtract the Math score from the CS score: 57.1 - 45.0 = 12.1. Therefore, GPT-4V's performance is 12.1 points higher in the Computer Science subject compared to the Math subject on the MMMU dat…

Agent原文：

> GPT-4V(ision) scores 65.3 on Art & Design and 48.4 on Science, a difference of 16.9 points, with Art & Design being higher.

实际Reasoner请求摘要：

```json
[{"pages": [16, 19], "full_images": 2, "crops": 1, "ocr_characters": 8151, "status": "insufficient"}, {"pages": [16, 19], "full_images": 2, "crops": 1, "ocr_characters": 8151, "status": "insufficient"}, {"pages": [5, 16, 19], "full_images": 3, "crops": 3, "ocr_characters": 14546, "status": "answerable"}]
```

分析：人工查看物理15与18确认Math=45.0、CS=57.1，差12.1。实际三次请求页集始终没有这两页，最后还答成Art & Design对Science。应追踪问题实体与待解决事实，而非只累计已读页数。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0183/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0183/attempt_0001.jsonl)

### #761：gold未命中不等于没有正确表

问题：How many strategies outperform the random selection baseline on CIFAR-10-LT?

- 标准答案：`4`
- e2e：score=1.0000，pred=`4`
- Agent：score=0.0000，pred=`6`；stop=`reasoner_answerable`，13轮。
- 标准证据页（物理0-based）：`[6]`；已读：`[5, 6, 14, 15, 18, 19, 20]`；最终选页：`[5, 18]`。

e2e原文（节选）：

> To determine the number of strategies that outperform random selection on CIFAR-10-LT, we must consult Table 1 on page 6. The table compares different querying strategies. For the CIFAR-10-LT dataset, the 'Random' strategy achieves an AUC score of 91.6±1.1 at 10% budget and 93.1±0.6 at 20% budget. We then compare these values with the other strategies: - Consistency: 91.7±0.9 (10%) and 93.2±0.2 (20%) - Both are better or equal. - VAAL: 92.6±0.2 (10%) and 93.7±0.4 (20%) - Both are better. - Margin: 91.7±0.9 (10%) and 93.2±0.2 (20%) - Both are better or equal. - Entropy: 65.6±15.6 (10%) and 86.4±0.2 (20%) - Both are worse. - Coreset: 91.5±0.4 (10%) and 93.6±0.2 (20%) - Both are better or equal…

Agent原文：

> Six strategies outperform the random selection baseline on CIFAR-10-LT: Consistency, VAAL, Margin, Entropy, Coreset, and BALD. This is evident from Table 7, where the AUC scores for these strategies are higher than those of the random selection baseline at various sampling ratios (1%, 5%, 10%, 20%, 30%, and 40%).

实际Reasoner请求摘要：

```json
[{"pages": [5, 18], "full_images": 2, "crops": 3, "ocr_characters": 11639, "status": "answerable"}]
```

分析：人工查看物理5确认它就是正确Table1且在Reasoner请求中。数据标注换算为物理6；Agent却根据另一Table7得6。应优先检查表/指标口径和干扰证据，而非自动归因为漏页。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0761/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0761/attempt_0001.jsonl)

### #818：问样本人数，却仅提供意见比例图

问题：How many U.S. adults express their opinions on if social media provides an accurate picture of how society feels about important issues?

- 标准答案：`4548`
- e2e：score=1.0000，pred=`4548`
- Agent：score=0.0000，pred=`74`；stop=`reasoner_answerable`，5轮。
- 标准证据页（物理0-based）：`[20, 28]`；已读：`[20]`；最终选页：`[20]`。

e2e原文（节选）：

> The methodology section on page 29 (image 29) states that the survey data is drawn from a panel wave conducted between May 29 and June 11, 2018, which included 4,594 respondents. Page 21 (image 21) presents a pie chart titled "Most think social media does not accurately reflect society," which shows the percentage of U.S. adults who believe social media content provides an accurate picture of societal feelings. The chart indicates that 74% believe it does not, while 25% believe it does. Therefore, the total number of U.S. adults who expressed an opinion on this topic is the sum of these two groups: 74% + 25% = 99%. To find the number of adults, we calculate 99% of the total sample size: 0.99…

Agent原文：

> 74% of U.S. adults express the opinion that social media does not provide an accurate picture of how society feels about important issues.

实际Reasoner请求摘要：

```json
[{"pages": [20], "full_images": 1, "crops": 1, "ocr_characters": 2790, "status": "answerable"}]
```

分析：物理20能看到意见比例，样本量还需要调查说明页。实际只提交20，输出74%。这是取数目标和证据覆盖同时缺失。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0818/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0818/attempt_0001.jsonl)

### #1034：同一证据多次insufficient，没有有效进展

问题：What is payables  turnover in FY2021 for Nike? Round your answer to two decimal places.

- 标准答案：`9.67`
- e2e：score=1.0000，pred=`9.67`
- Agent：score=0.0000，pred=`—`；stop=`max_rounds`，15轮。
- 标准证据页（物理0-based）：`[58, 60]`；已读：`[46, 58, 60]`；最终选页：`[58, 60]`。

e2e原文：

> The payables turnover ratio is calculated as Cost of Goods Sold (COGS) divided by Average Accounts Payable. From the 'CONSOLIDATED STATEMENTS OF INCOME' on page 57, the Cost of Sales for FY2021 is $24,576 million. From the 'CONSOLIDATED BALANCE SHEETS' on page 59, the Accounts Payable for FY2021 is $2,836 million and for FY2020 is $2,248 million. The average accounts payable is ($2,836 + $2,248) / 2 = $2,542 million. Therefore, the payables turnover is $24,576 / $2,542 ≈ 9.67.

Agent原文：

> —

实际Reasoner请求摘要：

```json
[{"pages": [58, 60], "full_images": 2, "crops": 2, "ocr_characters": 5097, "status": "insufficient"}, {"pages": [58, 60], "full_images": 2, "crops": 2, "ocr_characters": 5097, "status": "insufficient"}, {"pages": [58, 60], "full_images": 2, "crops": 2, "ocr_characters": 5097, "status": "insufficient"}, {"pages": [58, 60], "full_images": 2, "crops": 2, "ocr_characters": 5097, "status": "insufficient"}, {"pages": [58, 60], "full_images": 2, "crops": 2, "ocr_characters": 5097, "status": "insufficient"}, {"pages": [58, 60], "full_images": 2, "crops": 2, "ocr_characters": 5097, "status": "insufficient"}]
```

分析：实际六次Reasoner请求页集都为[58,60]。missing反复要求公式或某年AP；控制器继续搜索又提交同页。应记录缺的是公式还是操作数，并对重复无进展采取不同动作。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1034/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1034/attempt_0001.jsonl)

### #924：操作数正确，算术结论错误

问题：How much did the proportion of the favorable views of China among the American public decresed from 2005 to 2020?

- 标准答案：`21%`
- e2e：score=1.0000，pred=`21`
- Agent：score=0.0000，pred=`30`；stop=`reasoner_answerable`，3轮。
- 标准证据页（物理0-based）：`[3]`；已读：`[2, 3, 7]`；最终选页：`[3]`。

e2e原文：

> Based on the line chart titled "Unfavorable views of China reach new highs in U.S." on page 4, we can determine the change in favorable views of China from 2005 to 2020. - In 2005, the percentage of Americans with a favorable view of China was 43%. - In 2020, this percentage dropped to 22%. To find the decrease, we subtract the 2020 value from the 2005 value: 43% - 22% = 21%. Therefore, the proportion of favorable views of China decreased by 21 percentage points from 2005 to 2020.

Agent原文：

> The proportion of favorable views of China among the American public decreased by 30 percentage points from 2005 to 2020, dropping from 43% in 2005 to 22% in 2020.

实际Reasoner请求摘要：

```json
[{"pages": [3], "full_images": 1, "crops": 1, "ocr_characters": 1357, "status": "answerable"}]
```

分析：43−22=21，原文却写成30。两边都看到正确数据，Agent需要算术校验而不是更大上下文。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0924/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0924/attempt_0001.jsonl)

### #1080：子和算错导致财务净额错误

问题：what is the net fair value of derivative instruments for 3M in FY2018? Answer in million.

- 标准答案：`93`
- e2e：score=1.0000，pred=`93`
- Agent：score=0.0000，pred=`108`；stop=`reasoner_answerable`，4轮。
- 标准证据页（物理0-based）：`[103]`；已读：`[105, 106, 107]`；最终选页：`[106]`。

e2e原文：

> Based on the table titled "Location and Fair Value Amount of Derivative Instruments" on page 104 (image 104), we can find the net fair value of derivative instruments for the year 2018. 1. The table shows the total fair value amount for assets is $146 million. 2. The table shows the total fair value amount for liabilities is $53 million. 3. The net fair value is calculated by subtracting the liabilities from the assets: $146 million - $53 million = $93 million. Therefore, the net fair value of derivative instruments for 3M in FY2018 was $93 million.

Agent原文：

> The net fair value of derivative instruments for 3M in FY2018 is $108 million, calculated as $146 million in assets (127 + 19) minus $38 million in liabilities (22 + 31).

实际Reasoner请求摘要：

```json
[{"pages": [106], "full_images": 1, "crops": 1, "ocr_characters": 4935, "status": "answerable"}]
```

分析：127+19=146，22+31=53，净额93。Agent把负债子和写成38，最终108。原文中已有可用操作数，可直接用受限计算工具核验。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1080/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1080/attempt_0001.jsonl)

### #964：集合列对了，集合差仍选错

问题：What stages does high level lifecycle have more than basic lifecycle?

- 标准答案：`['concept', 'production']`
- e2e：score=1.0000，pred=`['Concept', 'Production']`
- Agent：score=0.0000，pred=`["Concept", "Inception"]`；stop=`reasoner_answerable`，2轮。
- 标准证据页（物理0-based）：`[9, 10]`；已读：`[9, 10]`；最终选页：`[9, 10]`。

e2e原文：

> Based on the provided slides, the High Level Lifecycle (Page 10) includes the following stages: Concept, Inception, Construction, Transition, and Production. The Basic/Agile Lifecycle (Page 11) includes the stages: Inception, Construction, and Transition. By comparing these two lists, the stages present in the High Level Lifecycle but not in the Basic/Agile Lifecycle are Concept and Production.

Agent原文：

> The high level lifecycle includes Concept, Inception, Construction, Transition, and Production. The basic lifecycle includes Inception, Construction, and Transition. Therefore, the high level lifecycle has Concept and Inception more than the basic lifecycle.

实际Reasoner请求摘要：

```json
[{"pages": [9, 10], "full_images": 2, "crops": 3, "ocr_characters": 770, "status": "answerable"}]
```

分析：原文的两组阶段集合已经足够推出Concept和Production。它却选Concept和Inception，说明取证成功后仍缺显式集合运算。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0964/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0964/attempt_0001.jsonl)

### #473：列举数量与结论不一致

问题：How many types of main menu functions are there?

- 标准答案：`11`
- e2e：score=1.0000，pred=`11`
- Agent：score=0.0000，pred=`10`；stop=`reasoner_answerable`，4轮。
- 标准证据页（物理0-based）：`[28]`；已读：`[28, 29]`；最终选页：`[28]`。

e2e原文：

> Page 29 of the manual displays a section titled "Main menu functions". By counting the distinct icons and their corresponding labels in this section, we can determine the number of functions. The listed functions are: Care, Beverages, Operating options, Information, Accounting, PIN rights, Timer, System, Language, Eco-mode, and USB. This amounts to a total of 11 main menu functions.

Agent原文：

> There are 10 types of main menu functions: Care, Beverages, Operating options, Information, Accounting, PIN rights, Timer, System, Language, Eco-mode, and USB.

实际Reasoner请求摘要：

```json
[{"pages": [28], "full_images": 1, "crops": 25, "ocr_characters": 3719, "status": "answerable"}]
```

分析：Care至USB共11项，原文却宣称10。可以通过结构化items数组与count字段的一致性检查发现。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0473/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0473/attempt_0001.jsonl)

### #547：表格已给出，行列条件匹配错误

问题：What's the AP50 value of DETR with L1 loss and without GIoU loss on the COCO validation set?

- 标准答案：`57.3`
- e2e：score=1.0000，pred=`57.3`
- Agent：score=0.0000，pred=`61.6`；stop=`reasoner_answerable`，6轮。
- 标准证据页（物理0-based）：`[12]`；已读：`[11, 12, 20]`；最终选页：`[12]`。

e2e原文：

> The answer is found in Table 4 on page 13, which details the effect of different loss components on the model's performance. The second row of the table corresponds to the model trained with only the L1 loss (indicated by a checkmark in the 'l1' column and an empty cell in the 'GIoU' column). Following this row to the 'AP50' column reveals the value 57.3.

Agent原文：

> 61.6

实际Reasoner请求摘要：

```json
[{"pages": [12], "full_images": 1, "crops": 4, "ocr_characters": 3119, "status": "answerable"}]
```

分析：e2e按L1有勾、GIoU无勾、AP50列取57.3；Agent实际提交同一目标页却取61.6。此类问题需要行列定位与表格读数校验，不能只优化检索召回。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0547/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0547/attempt_0001.jsonl)

### #588：邮箱中的小写l与大写I混淆

问题：what is the email id of the mtre laurent nahmiash ?

- 标准答案：`lnahmiash@infavocats.com`
- e2e：score=1.0000，pred=`lnahmiash@infavocats.com`
- Agent：score=0.0000，pred=`Inahmiash@infavocats.com`；stop=`reasoner_answerable`，3轮。
- 标准证据页（物理0-based）：`[13]`；已读：`[13, 14]`；最终选页：`[13, 14]`。

e2e原文：

> Based on the information provided in the document, the email id of Mtre Laurent Nahmiash is lnahmiash@infavocats.com. This can be found on page 14 and page 15 of the document.

Agent原文：

> The email id of Mtre Laurent Nahmiash is Inahmiash@infavocats.com, as shown in the document.

实际Reasoner请求摘要：

```json
[{"pages": [13, 14], "full_images": 2, "crops": 1, "ocr_characters": 1441, "status": "answerable"}]
```

分析：pred的首字符是I，标准是l，邮箱触发精确匹配。这是一个字符级辨识问题，需回看OCR与局部图；不宜泛化为所有字符串都用模糊匹配。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0588/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0588/attempt_0001.jsonl)

### #58：e2e原文已正确，最终抽取失败

问题：As of Q3 2015, are there more active Instagram users in Vietnam, or are there more fans for Appota's fan page? Directly answer 'Instagram' or 'Appota'.

- 标准答案：`Appota`
- e2e：score=0.0000，pred=`Not answerable`
- Agent：score=1.0000，pred=`Appota`；stop=`reasoner_answerable`，3轮。
- 标准证据页（物理0-based）：`[27, 32]`；已读：`[27, 32]`；最终选页：`[27, 32]`。

e2e原文：

> Appota

Agent原文：

> Appota

实际Reasoner请求摘要：

```json
[{"pages": [27, 32], "full_images": 2, "crops": 8, "ocr_characters": 1863, "status": "answerable"}]
```

分析：e2e原文只有Appota，Agent原文也只有Appota，e2e却抽成Not answerable。两边抽取组织差异是必须先控制的实验变量。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0058/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0058/attempt_0001.jsonl)

### #103：多页条件筛选补出漏项

问题：Which colleges/schools/institutes from NTU use year 3 students as cover page student model?

- 标准答案：`['College of Engineering', 'Nanyang Business School', 'National Institute of Education']`
- e2e：score=0.0000，pred=`['Nanyang Business School (NBS)', 'National Institute of Education (NIE)']`
- Agent：score=1.0000，pred=`["College of Engineering", "Nanyang Business School (NBS)", "National Institute of Education (NIE)"]`；stop=`reasoner_answerable`，10轮。
- 标准证据页（物理0-based）：`[9, 16, 17]`；已读：`[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]`；最终选页：`[9, 16, 17]`。

e2e原文：

> Based on the document, the following colleges/schools/institutes use a Year 3 student as the cover page model: - **Nanyang Business School (NBS)**: Page 17 features Wang Shi Lin Nicole, who is identified as a Business student in Year 3. - **National Institute of Education (NIE)**: Page 18 features Ruth Koh, who is identified as a Sport Science & Management student in Year 3.

Agent原文：

> Based on the provided documents, the following colleges/schools/institutes from NTU use a Year 3 student as the cover page model: 1. **College of Engineering**: The cover page features Stella Chua Zhi Yun, a Year 3 Mechanical Engineering student. 2. **Nanyang Business School (NBS)**: The cover page features Wang Shi Lin Nicole, a Year 3 Business student. 3. **National Institute of Education (NIE)**: The cover page features Ruth Koh, a Year 3 Sport Science & Management student.

实际Reasoner请求摘要：

```json
[{"pages": [9, 16, 17], "full_images": 3, "crops": 27, "ocr_characters": 17027, "status": "answerable"}]
```

分析：e2e只列NBS和NIE；Agent分别查看三个学生页，给出Year3机械工程学生所属College of Engineering。实际提交9/16/17，而非全文所有页。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0103/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0103/attempt_0001.jsonl)

### #106：目标餐桌图聚焦与crop的候选收益

问题：How many goblets appear in the figure showing Skyskraoeren?

- 标准答案：`7`
- e2e：score=0.0000，pred=`6`
- Agent：score=1.0000，pred=`7`；stop=`reasoner_answerable`，4轮。
- 标准证据页（物理0-based）：`[22]`；已读：`[22, 24]`；最终选页：`[22]`。

e2e原文：

> The figure on page 23 (image 23) displays a dining setup at Skyskraperen. By counting the wine glasses, we can identify 6 distinct goblets arranged on the table.

Agent原文：

> There are 7 goblets in the figure showing Skyskraperen: 4 on the left side of the table and 3 on the right side.

实际Reasoner请求摘要：

```json
[{"pages": [22], "full_images": 1, "crops": 7, "ocr_characters": 2358, "status": "answerable"}]
```

分析：人工查看目标页确有细小酒杯图案。Reasoner收到一页整图和7个可用crop，原文说明左4右3；e2e数6。只有同证据去掉crop的消融才能隔离其贡献。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0106/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0106/attempt_0001.jsonl)

### #540：小字图标：全文中看不清，局部上下文中识别出来

问题：In Figure 113, what's the name of the music app shown in the computer screen?

- 标准答案：`YouTube Music`
- e2e：score=0.0000，pred=`Not answerable`
- Agent：score=1.0000，pred=`YouTube Music`；stop=`reasoner_answerable`，4轮。
- 标准证据页（物理0-based）：`[142]`；已读：`[113, 142]`；最终选页：`[142]`。

e2e原文：

> Figure 113 on page 143 displays a computer desktop. The task is to identify the music app from the icons on the left side of the screen. The icons are not clearly labeled with text, and their appearance is generic. The document does not provide any information to identify the specific music application. Therefore, the question is not answerable from the provided content.

Agent原文：

> YouTube Music

实际Reasoner请求摘要：

```json
[{"pages": [142], "full_images": 1, "crops": 1, "ocr_characters": 1754, "status": "answerable"}]
```

分析：166页文档里，Locator先误读物理113，再经Figure113检索找到142；Reasoner仅接收1整图+1截图crop，回答YouTube Music。人工看图确认桌面左上图标名很小，OCR另提供YouTube相关文字，增益可能来自聚焦和OCR/crop组合。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0540/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0540/attempt_0001.jsonl)

### #607：跨页算术成功，同时e2e进入生成循环

问题：What is the sum of percentage of customers and employees said that Wi-Fi increases customer loyalty in hospitality in 2015?

- 标准答案：`122`
- e2e：score=0.0000，pred=`Not answerable`
- Agent：score=1.0000，pred=`122`；stop=`reasoner_answerable`，3轮。
- 标准证据页（物理0-based）：`[36, 37]`；已读：`[36, 37]`；最终选页：`[36, 37]`。

e2e原文（节选）：

> To answer this question, we need to find the percentage of customers and employees who believe Wi-Fi increases customer loyalty in the hospitality sector for the year 2015 and then sum these percentages.1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 13. 14. 15. 16. 17. 18. 19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35. 36. 37. 38. 39. 40. 41. 42. 43. 44. 45. 46. 47. 48. 49. 50. 51. 52. 53. 54. 55. 56. 57. 58. 59. 60. 61. 62. 63. 64. 65. 66. 67. 68. 69. 70. 71. 72. 73. 74. 75. 76. 77. 78. 79. 80. 81. 82. 83. 84. 85. 86. 87. 88. 89. 90. 91. 92. 93. 94. 95. 96. 97. 98. 99. 100. 101. 102. 103. 104. 105. 106. 107. 108. 109. 110. 111. 112. 113. 114. 115. 116. 117. 118. 119. 120. 121. …

Agent原文：

> 122%

实际Reasoner请求摘要：

```json
[{"pages": [36, 37], "full_images": 2, "crops": 2, "ocr_characters": 1405, "status": "answerable"}]
```

分析：人工查看36/37两页，Hospitality的customer和employee忠诚度指标均61%。Agent将两页一次提交后答122%。e2e未形成数值答案，重复编号到8192 tokens截断。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0607/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0607/attempt_0001.jsonl)

### #803：从相似文字中找到真正的目标阶段图

问题：What are the five steps of ValueEdge Insights?

- 标准答案：`['Plan', 'Build', 'Test', 'Deliver', 'Run']`
- e2e：score=0.0000，pred=`['Plan, build, and deliver digital product value streams.', 'Discover, visualize, and manage flow—from strategy to delivery.', 'Accelerate business decision-making with AI and analytics.', 'Integrate your commercial and open source development tools.', 'Align your business objectives with your development resources.']`
- Agent：score=1.0000，pred=`["Plan", "Build", "Test", "Deliver", "Run"]`；stop=`reasoner_answerable`，13轮。
- 标准证据页（物理0-based）：`[16]`；已读：`[2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]`；最终选页：`[16]`。

e2e原文：

> Based on the information provided in the document, the five steps of ValueEdge Insights are: 1. Plan, build, and deliver digital product value streams. 2. Discover, visualize, and manage flow—from strategy to delivery. 3. Accelerate business decision-making with AI and analytics. 4. Integrate your commercial and open source development tools. 5. Align your business objectives with your development resources. These steps are listed on page 8 (image 8) under the 'Platform overview' section.

Agent原文：

> The five steps of ValueEdge Insights are: Plan, Build, Test, Deliver, and Run.

实际Reasoner请求摘要：

```json
[{"pages": [16], "full_images": 1, "crops": 1, "ocr_characters": 586, "status": "answerable"}]
```

分析：e2e引用平台overview的五条宣传文字。Agent读过多处候选，最终提交16，得到Plan/Build/Test/Deliver/Run。实际输入还包含1个crop和OCR，不能把优势全部归因于多轮规划。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0803/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0803/attempt_0001.jsonl)

### #1059：允许从原始量推导，避免因无字面比率而弃答

问题：What is Cash Flow to Total Debt Ratio for ADBE In FY2015?

- 标准答案：`0.77`
- e2e：score=0.0000，pred=`Not answerable`
- Agent：score=1.0000，pred=`0.77`；stop=`reasoner_answerable`，3轮。
- 标准证据页（物理0-based）：`[62, 98]`；已读：`[62, 98]`；最终选页：`[62, 98]`。

e2e原文：

> The document provides the necessary components to calculate this ratio but does not state the ratio itself. On page 51, the 'Consolidated Statements of Cash Flows' shows that the 'Net cash provided by operating activities' for Fiscal 2015 was $1,469.5 million. On page 59, the 'Consolidated Balance Sheets' shows 'Debt and capital lease obligations, non-current' as $1,907,231 thousand (or $1,907.231 million) for November 27, 2015. While one could calculate the ratio ($1,469.5 / $1,907.231), the document does not explicitly state this calculated value. Therefore, the question is not answerable from the provided text.

Agent原文：

> The Cash Flow to Total Debt Ratio for ADBE in FY2015 is 0.77. This is calculated by dividing the Net cash provided by operating activities ($1,469,502 thousand) by the Total debt and capital lease obligations ($1,907,231 thousand).

实际Reasoner请求摘要：

```json
[{"pages": [62, 98], "full_images": 2, "crops": 2, "ocr_characters": 10276, "status": "answerable"}]
```

分析：Agent集中62/98，计算1469502÷1907231≈0.77049。e2e原文也已有这些量，却认为文档没有直接写比率就不能答。主要差异是可计算性判断，不是e2e完全没找到数值。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1059/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1059/attempt_0001.jsonl)

### #174：相同国家事实，e2e抽取成了数值

问题：Which country had the highest percentage of respondents for whom traditional values mean a lot?

- 标准答案：`Oman`
- e2e：score=0.0000，pred=`60`
- Agent：score=1.0000，pred=`Oman`；stop=`reasoner_answerable`，3轮。
- 标准证据页（物理0-based）：`[10]`；已读：`[9, 10]`；最终选页：`[9, 10]`。

e2e原文：

> Based on the chart 'VALUES AND BELIEFS BY COUNTRY' on page 11, the percentage of respondents who believe 'Traditional values mean a lot to me, and ought to be preserved for generations to come' is shown in red. Comparing the values for all listed countries, Oman has the highest percentage at 60%.

Agent原文：

> Oman had the highest percentage of respondents for whom traditional values mean a lot, at 60%.

实际Reasoner请求摘要：

```json
[{"pages": [9, 10], "full_images": 2, "crops": 2, "ocr_characters": 1499, "status": "answerable"}]
```

分析：e2e原文已经明确Oman60%，最终pred却为60。Agentpred为Oman。这不应成为“Agent找到e2e找不到的国家”的训练标签。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0174/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0174/attempt_0001.jsonl)

### #418：单位丢失造成的零分

问题：How many people in India were using a debit card when the e-commerce sales reached 11 billion?

- 标准答案：`399 million`
- e2e：score=0.0000，pred=`399`
- Agent：score=1.0000，pred=`399 million`；stop=`reasoner_answerable`，7轮。
- 标准证据页（物理0-based）：`[11, 14]`；已读：`[11, 14]`；最终选页：`[11, 14]`。

e2e原文（节选）：

> Based on the provided slides, we can determine the number of debit card users in India at the time e-commerce sales reached $11 billion. - First, we need to identify when e-commerce sales were $11 billion. The slide titled "SHOW ME THE MONEY, HONEY!" (Page 12) shows a bar chart of eCommerce Sales. For the year 2014, the total sales are indicated as $11 Billion. - Next, we look for the number of debit card users in India for the year 2014. The slide titled "PAYMENTS LANDSCAPE" (Page 15) contains a bar chart showing the "Number of Debit Card users in India (In millions)". For the year 2014, this value is listed as 399 million. Therefore, when e-commerce sales in India reached $11 billion in 20…

Agent原文：

> In 2014, when e-commerce sales reached $11 billion, there were 399 million debit card users in India.

实际Reasoner请求摘要：

```json
[{"pages": [11, 14], "full_images": 2, "crops": 3, "ocr_characters": 1247, "status": "answerable"}]
```

分析：两边都正确回答399 million。e2e只抽399，标准Str为399 million，单位不可丢。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0418/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0418/attempt_0001.jsonl)

### #531：等价时长换算与指定字符串表示

问题：What is the average session duration for desktop?

- 标准答案：`00:04:14`
- e2e：score=0.0000，pred=`254`
- Agent：score=1.0000，pred=`00:04:14`；stop=`reasoner_answerable`，4轮。
- 标准证据页（物理0-based）：`[13]`；已读：`[13]`；最终选页：`[13]`。

e2e原文：

> Based on the table on page 14, the average session duration for the 'desktop' device category is 00:04:14.

Agent原文：

> The average session duration for desktop is 00:04:14, as shown in the table under the 'Avg. Session Duration' column for the 'desktop' row.

实际Reasoner请求摘要：

```json
[{"pages": [13], "full_images": 1, "crops": 1, "ocr_characters": 1310, "status": "answerable"}]
```

分析：两边原文均为00:04:14，e2e抽取成254秒。信息本身没错，最终表示与基准不一致，应该统一抽取格式。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0531/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0531/attempt_0001.jsonl)

### #593：相同人名，头衔使ANLS结果不同

问题：Which officer verify and signed the complaint?

- 标准答案：`Special Agent Marc Silski`
- e2e：score=0.0000，pred=`Marc Silski`
- Agent：score=1.0000，pred=`Special Agent Marc Silski`；stop=`reasoner_answerable`，3轮。
- 标准证据页（物理0-based）：`[15]`；已读：`[15]`；最终选页：`[15]`。

e2e原文（节选）：

> Based on the information provided in the document, the complaint was verified and signed by Marc Silski. Page 16 of the document contains a section titled "VERIFICATION". In this section, an individual states, "I, Marc Silski, am a Special Agent of the Federal Bureau of Investigation ('FBI'). I have read the foregoing Complaint for Forfeiture, and declare under penalty of perjury of the laws of the United States of America that the facts contained therein are true to the best of my knowledge and belief...". This statement is followed by a signature line which is signed by "Special Agent Marc Silski" from the "Federal Bureau of Investigation". Therefore, Marc Silski is the officer who verifie…

Agent原文：

> Special Agent Marc Silski of the Federal Bureau of Investigation verified and signed the complaint on September 6, 2018.

实际Reasoner请求摘要：

```json
[{"pages": [15], "full_images": 1, "crops": 1, "ocr_characters": 977, "status": "answerable"}]
```

分析：e2e原文提到Special Agent Marc Silski，最后只抽Marc Silski，字符串长度差使ANLS为0。Agent保留头衔而满分。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0593/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0593/attempt_0001.jsonl)

### #777：同一动物，冗余词触发字符串零分

问题：What is the first animal, other than humans, shown in this guidebook?

- 标准答案：`panda`
- e2e：score=0.0000，pred=`A giant panda`
- Agent：score=1.0000，pred=`panda`；stop=`reasoner_answerable`，5轮。
- 标准证据页（物理0-based）：`[3]`；已读：`[0, 1, 2, 3, 4, 5]`；最终选页：`[3]`。

e2e原文：

> The guidebook's table of contents on page 6 (image 6) indicates that the 'Introduction' section begins on page 7. Reviewing the pages following the contents, page 10 (image 10) is the first page in the main body of the guide to feature an animal. The image on this page clearly shows a giant panda. Subsequent pages, like page 12 (image 12), also feature pandas, confirming it as the first animal depicted. The answer of the question is: A giant panda

Agent原文：

> The first animal shown in the guidebook, other than humans, is a panda.

实际Reasoner请求摘要：

```json
[{"pages": [3], "full_images": 1, "crops": 1, "ocr_characters": 65, "status": "answerable"}]
```

分析：e2e的A giant panda与标准panda语义一致，但编辑距离超过阈值。不能把这一题当作Agent视觉识别能力优于e2e的证据。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0777/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0777/attempt_0001.jsonl)

### #988：简称与全称的评价边界

问题：Where did Terry Tan study the foundation course before he progressed to NTU?

- 标准答案：`NTIC`
- e2e：score=0.0000，pred=`Nottingham Trent International College`
- Agent：score=1.0000，pred=`NTIC (Nottingham Trent International College)`；stop=`reasoner_answerable`，3轮。
- 标准证据页（物理0-based）：`[32]`；已读：`[32]`；最终选页：`[32]`。

e2e原文：

> Page 33 (image 33) features a testimonial from Terry Tan. The text below his photo states, "Before I progressed to NTU, I studied the foundation course at NTIC." The contact information box on the same page clarifies that NTIC stands for Nottingham Trent International College. Therefore, Terry Tan studied at Nottingham Trent International College.

Agent原文：

> Terry Tan studied the foundation course at NTIC (Nottingham Trent International College) before progressing to NTU.

实际Reasoner请求摘要：

```json
[{"pages": [32], "full_images": 1, "crops": 3, "ocr_characters": 2014, "status": "answerable"}]
```

分析：e2e原文明确NTIC全称，最后抽全称；Agent保留NTIC并附括号说明，官方清洗括号后匹配简称。应标记为评价表达差异。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0988/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0988/attempt_0001.jsonl)

### #12：最终满分，分项解释却无法加总

问题：How many charts and tables in this report are sourced from Annual totals of Pew Research Center survey data?

- 标准答案：`7`
- e2e：score=0.0000，pred=`5`
- Agent：score=1.0000，pred=`7`；stop=`reasoner_answerable`，9轮。
- 标准证据页（物理0-based）：`[2, 5, 15, 17, 18, 19, 21]`；已读：`[2, 5, 15, 17, 18, 19, 21]`；最终选页：`[2, 5, 15, 17, 19]`。

e2e原文（节选）：

> To answer this question, I will examine the source note for each chart and table in the document. - Page 3: The stacked area chart "Independents outnumber Republicans and Democrats..." is sourced from "Annual totals of Pew Research Center survey data". (Count: 1) - Page 4: The bar chart "Independents who do not lean to a party less likely to say they voted in 2018" is sourced from a "Survey of U.S. adults conducted Nov. 7-16, 2018 using the online American Trends Panel." This is not an annual total. - Page 6: The line chart "During his first two years in office, Trump's job rating..." is sourced from "Annual totals of Pew Research Center survey data". (Count: 2) - Page 8: The bar chart on bo…

Agent原文：

> There are 7 charts in this report sourced from Annual totals of Pew Research Center survey data: 1 on page 2, 4 on page 5, 3 on page 15, 1 on page 17, and 2 on page 19. There are no tables in the report.

实际Reasoner请求摘要：

```json
[{"pages": [2, 5, 15, 17, 19], "full_images": 5, "crops": 12, "ocr_characters": 9242, "status": "answerable"}]
```

分析：原文7与1+4+3+1+2=11矛盾，且漏掉提交未包含的两页。分数保留，但轨迹暂不作为可靠的全局计数示例。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0012/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0012/attempt_0001.jsonl)

### #345：开头结论与后文计算矛盾

问题：What is the percentage difference between the proportion of people who are very confident in Trump's ability to make wise decisions about immigration policy and those who are not confident at all?

- 标准答案：`16%`
- e2e：score=0.0000，pred=`18`
- Agent：score=1.0000，pred=`16`；stop=`reasoner_answerable`，4轮。
- 标准证据页（物理0-based）：`[6]`；已读：`[6]`；最终选页：`[6]`。

e2e原文（节选）：

> To answer this question, information from two separate pages must be integrated. First, on page 7 (image 7), the text states that regarding immigration policy, "45% who say they are not at all confident in him on this issue." Second, on page 8 (image 8), the bar chart titled "Republicans broadly confident in Trump on most issues..." shows the breakdown of confidence levels. For the category "Make wise decisions about immigration policy," the percentage of people who are "Very" confident is listed as 65% for Republicans/Lean Rep and 2% for Democrats/Lean Dem. The question asks for the overall percentage, which is not directly provided in a single number. However, we can calculate it from the …

Agent原文：

> The percentage difference is 13 percentage points. The proportion of people who are very confident is 29%, and the proportion of people who are not confident at all is 45%. The difference is 45% - 29% = 16%. However, the question asks for the difference between the proportion of people who are very confident and those who are not confident at all. The difference is 45% - 29% = 16%. Therefore, the percentage difference is 16 percentage points.

实际Reasoner请求摘要：

```json
[{"pages": [6], "full_images": 1, "crops": 1, "ocr_characters": 2143, "status": "answerable"}]
```

分析：同一条回答先说13，后又两次写45−29=16；抽取选中16而得分。建议保留结果奖励，同时标记一致性问题。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0345/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0345/attempt_0001.jsonl)

### #568：最终差值对，子和错

问题：What is the percentage difference between the sum of women who have received a STEM degree and are employed in their field, and the sum of men with the same status in the survey of Pew Research Center, January 2018?

- 标准答案：`15%`
- e2e：score=0.0000，pred=`21`
- Agent：score=1.0000，pred=`15`；stop=`reasoner_answerable`，5轮。
- 标准证据页（物理0-based）：`[43]`；已读：`[17, 43]`；最终选页：`[17, 43]`。

e2e原文：

> The table on page 18 (image 18) provides data on college-educated workers employed in jobs related to their bachelor's degree field. For 'Computer major working in computers', 38% of women are employed in this field. For 'Engineering major working in engineering', 24% of women are employed in this field. The sum for women is 38% + 24% = 62%. For men, the corresponding figures are 53% for computer majors and 30% for engineering majors. The sum for men is 53% + 30% = 83%. The difference between the two sums is 83% - 62% = 21%.

Agent原文：

> The percentage difference is 15 percentage points. The sum for women is 114% (69% + 38% + 24% + 5% + 5% + 8%), and the sum for men is 99% (61% + 53% + 30% + 5% + 5% + 10%).

实际Reasoner请求摘要：

```json
[{"pages": [17, 43], "full_images": 2, "crops": 2, "ocr_characters": 6075, "status": "answerable"}]
```

分析：列出的女性加数和149、男性164，差值绝对值15；文字却称女性114、男性99。最终分数正确不能证明中间推理正确。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0568/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0568/attempt_0001.jsonl)

### #621：计数值正确，所数对象类别不对

问题：How many website URLs are contained in the slides?

- 标准答案：`3`
- e2e：score=0.0000，pred=`2`
- Agent：score=1.0000，pred=`3`；stop=`reasoner_answerable`，9轮。
- 标准证据页（物理0-based）：`[41, 42, 47]`；已读：`[41, 42, 47]`；最终选页：`[41, 42, 47]`。

e2e原文：

> Based on the provided slides, there are 2 website URLs mentioned. They can be found on the final slide (Page 48). The first URL is for the presenter's email address: russell.james@ttu.edu. The second URL is a link to an outline for a behavioral economics series: http://www.slideshare.net/rnja8c/outline-for-behavioral-economics-course-component. Although other slides contain text that looks like a URL (e.g., on Page 42 and 43), they are presented as hyperlinks within the context of the presentation's content rather than as standalone URLs for the user to visit. Therefore, counting only the distinct, functional-looking URLs on the final slide, the total is 2.

Agent原文：

> There are 3 website URLs in the slides: 1. http://sports.espn.go.com/espn/e60/news/story?id=4388790 (appears on pages 41 and 42), 2. Bryce Harper 1.wmv (page 41), and 3. Bryce Harper 2.wmv (page 42).

实际Reasoner请求摘要：

```json
[{"pages": [41, 42, 47], "full_images": 3, "crops": 3, "ocr_characters": 2279, "status": "answerable"}]
```

分析：把两个.wmv文件名算成website URLs。仅靠最终数字命中筛正样本会把对象分类错误纳入训练。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0621/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0621/attempt_0001.jsonl)

### #832：答案开头说7，枚举到10

问题：how many LEARNING OUTCOMES should be ANSWERed in UNIT 8?

- 标准答案：`10`
- e2e：score=0.0000，pred=`9`
- Agent：score=1.0000，pred=`10`；stop=`reasoner_answerable`，4轮。
- 标准证据页（物理0-based）：`[5, 6]`；已读：`[5, 6]`；最终选页：`[5, 6]`。

e2e原文：

> Based on the information provided in the document, we can determine the number of learning outcomes for Unit 8. - Page 6 introduces "UNIT 8: Managing Human Resources and Labor Relations" and lists the first two learning outcomes under the heading "SWBAT TO ANSWER THESE LEARNING OUTCOMES:". - Page 7 continues the list of learning outcomes for Unit 8. - By counting the bullet points under this heading across both pages, we can find the total number. There are 2 learning outcomes listed on Page 6 and 7 learning outcomes listed on Page 7. Therefore, there are a total of 9 learning outcomes to be answered in Unit 8.

Agent原文：

> There are 7 learning outcomes to be answered in Unit 8: 1) What is the human resource management process, and how are human resource needs determined? 2) How do firms recruit applicants? 3) How do firms select qualified applicants? 4) What types of training and development do organizations offer their employees? 5) How are performance appraisals used to evaluate employee performance? 6) What are the types of compensation and methods for paying workers? 7) What is a labor union and how is it organized, what is collective bargaining, and what are some of the key negotiation issues? 8) How are grievances between management and labor resolved, and what tactics are used to force a contract settlement? 9) What are the key laws and federal agencies affecting human resource management and labor relations? 10) What trends and issues are affecting human resource management and labor relations?

实际Reasoner请求摘要：

```json
[{"pages": [5, 6], "full_images": 2, "crops": 0, "ocr_characters": 10030, "status": "answerable"}]
```

分析：抽取器从10条枚举中取得正确10；解释首句仍错误。这种计数一致性应可自动检查。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0832/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0832/attempt_0001.jsonl)

### #927：计算正确但首句残留错误数字

问题：Assume that after July 2020, the percentage of Republicans who believe "China is an enemy of the U.S." increased by 10 percentage points, while the percentage of Democrats remained the same.  What would be the difference in the percentage between Republicans and Democrats at that time?

- 标准答案：`29%`
- e2e：score=0.0000，pred=`Not answerable`
- Agent：score=1.0000，pred=`29`；stop=`reasoner_answerable`，3轮。
- 标准证据页（物理0-based）：`[6]`；已读：`[16]`；最终选页：`[16]`。

e2e原文（节选）：

> Based on the information provided in the document, this question is not answerable. Page 7 (image 7) shows that in the July 2020 survey, 82% of Republicans and 54% of Democrats believed "China has done a bad job handling the coronavirus outbreak," but it does not provide a direct percentage for the belief that "China is an enemy of the U.S." for either party in that specific survey. While page 16 (image 16) shows that 26% of Republicans and 16% of Democrats saw China as an enemy in 2020, the document does not link this specific question to the July 2020 data point used for other metrics. Therefore, we cannot establish the baseline percentage for the July 2020 survey to perform the hypothetic…

Agent原文：

> The difference would be 19 percentage points. In 2020, 38% of Republicans saw China as an enemy. Adding 10 percentage points gives 48%. Democrats remained at 19%, so the difference is 48% - 19% = 29 percentage points.

实际Reasoner请求摘要：

```json
[{"pages": [16], "full_images": 1, "crops": 2, "ocr_characters": 2423, "status": "answerable"}]
```

分析：正文开头19与末尾48−19=29矛盾，最终抽取29。所选物理16与标注6不一致，还提示标准页可能有问题。

[Agent逐题结果](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0927/result.json) · [原始调用轨迹](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0927/attempt_0001.jsonl)

## 9. 完整清单：e2e满分、Agent零分（129题）

以下是逐题输出比较和第一层诊断；页覆盖只作线索，具体因果以原文/轨迹和必要消融为准。长文本用省略号截短，完整值见CSV/JSON。

| 编号 | 问题 | 标准答案 | e2e得分 / pred | Agent得分 / pred | 诊断 |
| ---: | --- | --- | --- | --- | --- |
| [#5](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0005/result.json) | According to this report, among the Hispanics shaped by four current personal financial situation, which situation involves the highest percentage that expects their future financial situation to get a lot worse? | Poor Financial Condition | 1.0000 / Poor financial condition | 0.0000 / 8% | 原文已给正确群体poor financial condition，抽取成数值8%。 |
| [#16](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0016/result.json) | How does this document define the law of good gestalt? | Elements of objects tend to be perceptually grouped together if they form a pattern that is regular, simple, and orderly. | 1.0000 / "Elements of objects tend to be perceptually grouped together if they form a pattern that is regular, simple, and orderly." | 0.0000 / The document defines the Law of good Gestalt as the principle that elements of objects tend to be perceptually grouped together if they form… | 正确定义混入长解释，抽取器整段保留，ANLS归零。 |
| [#25](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0025/result.json) | How many organisations does this report introduce in detail (at least one paragraph) ? | 19 | 1.0000 / 19 | 0.0000 / 14 | 已提交全部标注页，组织枚举/计数14而标准19；需检查漏项与计数口径。 |
| [#28](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0028/result.json) | What is the web address in the orange box in page 47 that gives the Wiki Query Service code? | https://w.wiki/6gb | 1.0000 / https://w.wiki/6gb | 0.0000 / — | 已读目标页46，但Locator thinking截断，未提交Reasoner。 |
| [#36](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0036/result.json) | How many human quotes are provided with source in this document? | 5 | 1.0000 / 5 | 0.0000 / — | 已读全标注页；6次search、9次read，Reasoner请求为0，预算耗尽。 |
| [#44](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0044/result.json) | According to the flowchart "Analytics Value Chain", what comes between Reporting and Action? | Analysis | 1.0000 / Analysis | 0.0000 / Not answerable | Reasoner原文就是正确的Analysis，抽取器改成Not answerable。 |
| [#76](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0076/result.json) | In the figure that has a tree shape, what is the name of the branch that has the least leafs? | Pre-training | 1.0000 / Pre-training | 0.0000 / — | 读到目标树图后，thinking大量重复RAG，8192 tokens耗尽。 |
| [#90](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0090/result.json) | Under the pro mode to shoot, what is the function of the icon on right hand side of the icon that select a focus mode? | set white balance | 1.0000 / Set white balance | 0.0000 / adjust the color cast by selecting a mode (e.g., for bright daylight or overcast conditions) or to adjust the color temperature | 原文描述调整色偏/色温，与白平衡语义接近，表述未归一而零分；不宜只按gold未命中归罪定位。 |
| [#93](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0093/result.json) | What devices other than phone are introduced for setting device+? | ['vision', 'bluetooth device'] | 1.0000 / ['Vision', 'Bluetooth device'] | 0.0000 / Vision and Bluetooth device | 原文包含正确两个设备，抽取未转成列表，官方按单元素比较而零分。 |
| [#115](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0115/result.json) | Please list all countries whose power and influence respondents would like to see diminished in the results of this report on Questions Q41a-t. | ['China', 'North Korea', 'Russia', 'Iran'] | 1.0000 / ["China", "Iran", "North Korea", "Russia"] | 0.0000 / ["Russia", "North Korea", "Iran"] | 提交含目标页，输出列表漏China。 |
| [#119](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0119/result.json) | List all countries, except U.S. and German, that are mentioned in some illustrated surveys of this report. Note that we do not count EU as a single country. | ['UK', 'Israel', 'China', 'Canada', 'Mexico', 'Japan', 'France', 'Austria', 'Russia'] | 1.0000 / ['UK', 'China', 'Canada', 'Israel', 'Mexico', 'France', 'Russia', 'Austria', 'Japan'] | 0.0000 / — | 已经读全标注页，搜索/读页用尽轮数，未向Reasoner发请求。 |
| [#120](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0120/result.json) | Among people who age 30-49, what is the difference of percentage value between Americen and Germen having positive view on their bilateral relationship? | 37% | 1.0000 / 37 | 0.0000 / 1 | 已提交目标页，但年龄组/国家数值读错为72与71，结果1而非37。 |
| [#123](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0123/result.json) | According to this report, what percentage of Germany respondent view China or Japan as the top economic power? Give me a percantage value. | 59% | 1.0000 / 59 | 0.0000 / 6 | 目标页已提交，返回6而不是59，图表读数/范围不符。 |
| [#126](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0126/result.json) | What degree does LEBOUR have? | M.A. | 1.0000 / M.A. | 0.0000 / M.A., F.G.S. | 输出M.A., F.G.S.，包含非所问学位的额外头衔，字符串评分零分。 |
| [#136](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0136/result.json) | How many regulations of the HSCA 2008 are breached in all according to this report? | 7 | 1.0000 / 7 | 0.0000 / 6 | 标注页齐备，法规条目聚合/去重为6而非7。 |
| [#143](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0143/result.json) | List the no. of chapters including research highlight. The answer shall be formatted as a list, for example, ["Chapter 5", "Chapter 8"]. | ['Chapter 2', 'Chapter 3', 'Chapter 5', 'Chapter 6'] | 1.0000 / ["Chapter 2", "Chapter 3", "Chapter 5", "Chapter 6"] | 0.0000 / — | answer数组被接口拒绝；数组内容也需判分，不能直接补正确。 |
| [#167](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0167/result.json) | How many percentage respondents in this survey access to internet few times a week or more? | 14% | 1.0000 / 14 | 0.0000 / 7 | 只读物理38，未读标注39/40，结果7而非14；定位覆盖缺口。 |
| [#179](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0179/result.json) | How many samples in MMMU belong to sociology subject? | 287 | 1.0000 / 287 | 0.0000 / 252 | 未读标注2，提交物理17，人数252而非287；表格选择/定位缺口。 |
| [#181](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0181/result.json) | How many Economics-related error cases fall into the Error Category of Perceptual Error? | 2 | 1.0000 / 2 | 0.0000 / 1 | 目标页在证据中，错误类别/案例聚合计为1而非2。 |
| [#183](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0183/result.json) | According to this paper, What's the GPT-4V's performance difference between CS subject and math subject on MMMU datasets? | 12.1% | 1.0000 / 12.1 | 0.0000 / Not answerable | 正确Math/CS表页15/18已读，却从未提交；答成Art & Design与Science。 |
| [#186](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0186/result.json) | What is the performance of filter-then-rerank methods (w.o. ensemble) on 50-shot TACREV dataset? | 72.3% | 1.0000 / 72.3 | 0.0000 / 72.3(1.4) with InstructGPT and 72.3(0.3) with GPT-4 | 正确均值72.3被抽成含标准差和模型名的长串，Float解析失败。 |
| [#200](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0200/result.json) | How many figures are shown in this guidebook to teach users "Two-finger scrolling" tips? | 1 | 1.0000 / 1 | 0.0000 / 3 | 手势图的范围/类型混淆，连同邻近图计为3而非1。 |
| [#221](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0221/result.json) | What is the percentage gap between male 65+ age group who use internet and broadband at home in the Pew Research Center’s Internet Project July 18-September 30, 2013 tracking survey? Please write the answer into float format. | 12% | 1.0000 / 12.0 | 0.0000 / 10.0 | 只提交15/16，缺少标注6/7；人口组/指标表未对齐。 |
| [#223](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0223/result.json) | Among the Higher-income seniors, what are the percentage of them go online, has smartphone phone, and own a tablet computer? Please write the answer in the list format and in descend order,e.g., ["9%","8%"] in the Pew Research Cen… | ['90%', '42%', '39%'] | 1.0000 / ["90%","42%","39%"] | 0.0000 / — | answer数组触发类型失败，同时标注页8未读。 |
| [#236](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0236/result.json) | Among all the pictures, in which page the picture contains the most number of people? Give me the page no. as an integer. | 9 | 1.0000 / 9 | 0.0000 / 8 | 返回内部页索引8，标准要求人类页号9。 |
| [#248](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0248/result.json) | What is the sum of the number of cars in the figures on page 2 and the number of bars in the figures on page 4? Write the answer in int format. | 6 | 1.0000 / 6 | 0.0000 / — | 相同目标页已提交，Reasoner两次仍称缺汽车/柱子数量，后续触顶。 |
| [#258](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0258/result.json) | Comparing to the Telecom Operators in 2014 and 2013-2014, What are the Operators that are not in common? Please list the names in list format, e.g., ["a","b"] | ['3', 'Esia', 'Smartfren'] | 1.0000 / ["3", "Smartfren", "Esia"] | 0.0000 / — | answer数组被拒绝；所列运营商本身也与标准有差异。 |
| [#259](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0259/result.json) | What are the Online Games native major internet companies and the Telecom Operator name of the second largest Prepaid ARPU in 2008? Please list the answer in list with reverse alphabetical order. | ['XL', 'Kotakgame.com', 'GameQQ.net'] | 1.0000 / ["XL", "Kotakgame.com", "GameQQ.net"] | 0.0000 / ["Telkomsel", "Kotakgame.com", "GameQQ.net"] | 在线游戏公司基本对，但ARPU第二名错答Telkomsel，非XL。 |
| [#265](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0265/result.json) | How many lines go through the Union Square/Market Street station in San Francisco in the slides? | 6 | 1.0000 / 6 | 0.0000 / — | 已读目标地图页14，Locator thinking耗尽生成预算。 |
| [#278](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0278/result.json) | What is the sum of the total number of paid search's conversions in the year of 2007, 2008 and the number of green bars in the heroes happen here launch? | 4 | 1.0000 / 4 | 0.0000 / 3.0 | 未读标注94/106，提交107/131；还漏加绿色柱子的数量。 |
| [#281](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0281/result.json) | How many numbers with blue color are mentioned in the example figure of question-guidue reasoning? | 2 | 1.0000 / 2 | 0.0000 / [2, 212, 1000] | 问题问蓝色数字的个数，输出数字值列表，目标类型不匹配。 |
| [#295](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0295/result.json) | What is the highest hit rate number in the dataset for the personality located furthest to the left in the distribution of personality traits figure? | 0.833 | 1.0000 / 0.833 | 0.0000 / 0.970 | 页已提交，但人格特征/数据集/指标交叉匹配错误。 |
| [#302](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0302/result.json) | How many USD dollars will the author spend if the annotators finish annotating all of claims of the authors' proposed dataset? Please write the answer in float format. | 453.25 | 1.0000 / 453.25 | 0.0000 / 0.45425 | 已提交目标页，费用单位/数量级换算错误，0.45425而非453.25。 |
| [#304](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0304/result.json) | How many figures are there in the Appendix in the paper? Please write down the answer into int format. | 10 | 1.0000 / 10 | 0.0000 / — | 大量读页后触顶，未提交Reasoner；标注页15尚未读。 |
| [#321](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0321/result.json) | How many "subjects" with black color does the figure in Page 2 contains? | 0 | 1.0000 / 0 | 0.0000 / — | 页引用与标注有差异，Reasoner未形成答案后触顶，需人工核对页映射。 |
| [#331](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0331/result.json) | How many dogs and cats are there in page 17? Please write the answer in list format, e.g., ["1","2"] | ['0', '0'] | 1.0000 / ["0", "0"] | 0.0000 / — | 多次Reasoner把犬猫数量问题返回insufficient；未形成合法的零值列表答案，随后触顶。 |
| [#338](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0338/result.json) | How many NUSSU constituent clubs and NUSSU committees are there respectively? Please write the answer in list format. | ['15', '9'] | 1.0000 / [15, 9] | 0.0000 / — | 返回JSON列表触发类型拒绝，而非JSON解码失败。 |
| [#355](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0355/result.json) | In the United States, which group has the highest proportion of people who believe that the U.S. will have less influence in world affairs after the coronavirus outbreak compared to before the outbreak? | Liberal | 1.0000 / Liberal | 0.0000 / Democrats | 同证据中的党派与意识形态群体混用，Democrats而非Liberal。 |
| [#377](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0377/result.json) | What are the years specified in the pie chart of the document? Format all of the years in a list like ['2006', '2007', '2011'] | ['1981', '1982', '2001', '2002'] | 1.0000 / ['1981', '1982', '2001', '2002'] | 0.0000 / ["1981-82", "2001-02"] | 把1981-82/2001-02两个区间保留成两项，未列出所问四个年份。 |
| [#403](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0403/result.json) | Which chapters are included in the slide? Your answer should be formatted as a list like ['Chapter 1', 'Chapter 3']. | ['Chapter 8', 'Chapter 9', 'Chapter 10'] | 1.0000 / ['Chapter 8', 'Chapter 9', 'Chapter 10'] | 0.0000 / — | 返回章节数组被严格str校验拒绝。 |
| [#409](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0409/result.json) | Which stages of cell division are shown on slides 12 and 14? | ['Interphase', 'Prophase', 'Prometaphase', 'Metaphase', 'Anaphase', 'Telophase and cytokinesis'] | 1.0000 / ['Interphase', 'Prophase', 'Prometaphase', 'Metaphase', 'Anaphase', 'Telophase and Cytokinesis'] | 0.0000 / ["Interphase", "Cytokinesis"] | 把问题中slides12/14直接当0-based12/14，读成后两页。 |
| [#423](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0423/result.json) | How many 'Pioneers' and people on the cover of the slides are there in total in the slide? | 12 | 1.0000 / 12 | 0.0000 / 16 | 人物计数与叙述不一致，4+11也不等于最终16。 |
| [#429](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0429/result.json) | How many times does the image of a bear appear in the PPT? | 2 | 1.0000 / 2 | 0.0000 / — | 已经找到两处熊图，仍循环枚举计划，thinking截断。 |
| [#430](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0430/result.json) | How many times does the image of a tiger appear in the PPT? | 0 | 1.0000 / 0 | 0.0000 / — | 零只老虎的可回答题；未完成全局排查前thinking截断，标注页为空。 |
| [#431](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0431/result.json) | How many layers does the neural network mentioned in the PPT for distinguishing faces have? | 5 | 1.0000 / 5 | 0.0000 / 4 | 原文列输入+3隐藏+输出却称4，且参考页不全；层计数内部不一致。 |
| [#443](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0443/result.json) | How many years after The Chapel Bridge was The Acropolis Museum completed? Return me an integer. | 676 | 1.0000 / 676 | 0.0000 / — | 相同页反复insufficient，始终找不到完成年份，触顶。 |
| [#451](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0451/result.json) | In the 4-shot classification task, how much higher is the top-5 average accuracy of RAR (LLaVA1.5) compared to the top-1 accuracy? | 18.9% | 1.0000 / 18.9 | 0.0000 / 1.1 | 缺少目标页23且引用其他表，4-shot/top-5/top-1条件未对齐。 |
| [#453](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0453/result.json) | How many green bars appear in Figure 1? | 6 | 1.0000 / 6 | 0.0000 / 18 | 目标图已提交，绿色柱子6条却回答18，与自己每任务1条的描述不一致。 |
| [#459](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0459/result.json) | How many pie charts are there in the paper? | 5 | 1.0000 / 5 | 0.0000 / — | 已读饼图标注页5/11，但从未提交Reasoner，触顶。 |
| [#473](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0473/result.json) | How many types of main menu functions are there? | 11 | 1.0000 / 11 | 0.0000 / 10 | 列出11个菜单项但总数写10。 |
| [#482](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0482/result.json) | How many cars appear on page three? | 4 | 1.0000 / 4 | 0.0000 / 5 | 问题第三页读成物理3而非2，汽车数随之错误。 |
| [#483](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0483/result.json) | What technology does the car’s Wi-Fi Connect use? | 4G | 1.0000 / 4G | 0.0000 / 4G connectivity, specifically an AT&T Hotspot | 4G后保留长解释，未抽出简短技术名称，ANLS归零。 |
| [#484](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0484/result.json) | How many types of wheels are introduced in total? | 7 | 1.0000 / 7 | 0.0000 / 10 | 未读目标轮毂页13，输出跨页混合且有重复型号。 |
| [#493](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0493/result.json) | What animals appear on page nine? Enumerate them in a list. | ['dog', 'cat'] | 1.0000 / ['cat', 'dog'] | 0.0000 / ["A cat"] | 问题第九页读成物理9而非8，漏狗、只答猫。 |
| [#497](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0497/result.json) | What's the average value of all orange bars in the chart about the percentage of U.S. adults with Hispanic ancestry who self-identify as Hispanic or Non-Hispanic? | 21 | 1.0000 / 21.0 | 0.0000 / 79 | 把橙色对应人群读反，算出79而标准21，叙述又出现81.5。 |
| [#498](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0498/result.json) | According to the 2015 National Survey of Latinos details, which group of Latinos has the largest sample size among the 1500 samples in total? | Foreign born (excl. PR) | 1.0000 / Foreign born (excl. PR) | 0.0000 / 795 | 原文包含正确Foreign born (excl. PR)，抽取成样本量795。 |
| [#502](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0502/result.json) | Which Hispanic origin group in the United States is more likely to have a positive impression of capitalism and be dissatisfied with the nation's direction? | Cubans | 1.0000 / Cubans | 0.0000 / Cubans and South Americans | 双条件交集错误，多答South Americans。 |
| [#511](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0511/result.json) | What is the Word written in Orange box on page 17? | Start new search | 1.0000 / Start new search | 0.0000 / Not answerable | 原文Start new search已正确，抽取成Not answerable。 |
| [#525](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0525/result.json) | In 2013, what is the total percentage of online adult users in the "White, Non-Hispanic" and "Hispanic" categories on the platform whose metrics include Followers, Tweet Clicks, Video Views, Tweet Reach, Engagement Rate, Completio… | 32 | 1.0000 / 32 | 0.0000 / 32% | 原文32%与目标32相容，Int解析不接受带百分号的pred。 |
| [#529](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0529/result.json) | Which metrics are NOT included in LinkedIn Metrics: Post Reach, Post Types, Video Views? | ['Post Reach', 'Video Views'] | 1.0000 / ["Post Reach", "Video Views"] | 0.0000 / Post Reach, Post Types, and Video Views | 自己列出的包含项Post Types又被列入不包含项，集合差矛盾。 |
| [#534](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0534/result.json) | Which of the following company logos do not appear in the slides? Facebook, AWS, Cargo and Manta. | AWS | 1.0000 / AWS | 0.0000 / ['AWS', 'Manta'] | 只读7未读标注1，多判Manta缺席。 |
| [#547](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0547/result.json) | What's the AP50 value of DETR with L1 loss and without GIoU loss on the COCO validation set? | 57.3 | 1.0000 / 57.3 | 0.0000 / 61.6 | 正确目标表已提交，L1/GIoU行和AP50列匹配错误，61.6而非57.3。 |
| [#549](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0549/result.json) | According to Fig 10 architecture of DETR’s transformer, what is the colour of the Multi-Head Self-Attention layer? | Red | 1.0000 / Red | 0.0000 / pink | 同一图的颜色答pink而标准Red；可能涉及色彩命名边界，需图像核对。 |
| [#562](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0562/result.json) | In 2015, how many percentage of surveyed adults, Republicans, and Democrats believed that the goverment was doing very/fairly well in reducing the threat of terrorism? Please write the answer in list format, e.g., ["3","2"] | ['46', '27', '64'] | 1.0000 / [46, 27, 64] | 0.0000 / — | 正确列表46/27/64被answer必须str的接口拒绝。 |
| [#564](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0564/result.json) | In 2015, among all those that are considered as bigger threats to well-being of the United States compared to August 2014, which one has the biggest R-D difference? | Iran's nuclear program | 1.0000 / Iran's nuclear program | 0.0000 / 27 | 原文正确项目Iran核计划被抽成其差值27。 |
| [#581](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0581/result.json) | What page has an eight-pointed snowflake image? Give me an integer. | 1 | 1.0000 / 1 | 0.0000 / 0 | 把内部物理索引0作为对外页码，标准为1。 |
| [#587](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0587/result.json) | What type of court is noted on the cover page? | SUPERIOR COURT | 1.0000 / Superior Court (Class Action) | 0.0000 / Superior Court (Class Action Division) of the Province of Quebec, District of Montreal | 原文含Superior Court，但过长的分庭/地名表述导致ANLS零分。 |
| [#588](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0588/result.json) | what is the email id of the mtre laurent nahmiash ? | lnahmiash@infavocats.com | 1.0000 / lnahmiash@infavocats.com | 0.0000 / Inahmiash@infavocats.com | 电子邮箱首字母把小写l识别为大写I，精确匹配失败。 |
| [#591](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0591/result.json) | Among the 26 Exhibits from P-1 to P-26, which Exhibits are not mentioned in the document? The answer should be formatted as a list, for example ["P-1","P-2"] | ['P-15', 'P-16', 'P-17', 'P-18'] | 1.0000 / ["P-15", "P-16", "P-17", "P-18"] | 0.0000 / — | 全局缺失展品的排查未完成，触顶；无gold页可供定位判断。 |
| [#599](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0599/result.json) | What are the colors of this document? List all the colors in the list format and by alphabetical, for example ["black", "white"] | ['black', 'blue', 'white'] | 1.0000 / ["black", "blue", "white"] | 0.0000 / ["black", "white"] | 全局颜色枚举漏blue，标注页为空，不能据此判定位命中。 |
| [#605](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0605/result.json) | Besides the two companies on page 42, what is the contact phone number of another company? Please write in string format. | +1-615-591-2955 | 1.0000 / +1-615-591-2955 | 0.0000 / Not answerable | 已读41却未提交，只给40/42，最终误报没有联系电话。 |
| [#610](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0610/result.json) | What percentage of the second largest demographic segment of respondents reported not conducting promotions to customers over Wi-Fi in 2015? | 85% | 1.0000 / 85 | 0.0000 / 89 | 已提交标注页，第二大群体/对应百分比匹配错误。 |
| [#624](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0624/result.json) | How many pictures of musical instruments are contained in the slides? | 2 | 1.0000 / 2 | 0.0000 / — | 已读乐器图标注页但持续扫页，未提交Reasoner，触顶。 |
| [#630](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0630/result.json) | When using GPT-4 as the base language model, how many datasets does Logic-LM (without self-refinement) outperform the two baseline models in? | 4 | 1.0000 / 4 | 0.0000 / 5 | 目标表已提交，without self-refinement与基线比较条件未落实。 |
| [#636](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0636/result.json) | How many complete in-context examples are shown in the figure at the top of page 6985? Please write down the answer with integer format. | 1 | 1.0000 / 1 | 0.0000 / 0 | 指定图所在页已提交，但完整示例数判0而非1。 |
| [#638](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0638/result.json) | Which type of semantic errors has the lowest total percentages in HOVER dataset? | Subtask | 1.0000 / Subtask | 0.0000 / Syntax error | 问semantic errors却比较Syntax error，类别范围错误。 |
| [#642](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0642/result.json) | Among the three correction strategies: training-time correction, generation-time correction, and post-hoc correction, which one has the most representative papers in the survey? | post-hoc correction | 1.0000 / Post-hoc correction | 0.0000 / Training-Time Correction | 自述三类16/17/33，却把16对应类选为最多。 |
| [#644](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0644/result.json) | How many strategies are plotted for training-time correction, generation-time correction, and post-hoc correction, respectively? Write the answer in the list format, e.g., ["1","2"] | ['3', '2', '3'] | 1.0000 / ["3", "2", "3"] | 0.0000 / ["14", "2", "3"] | 部分标注页未读且混淆策略计数，14/2/3而非3/2/3。 |
| [#649](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0649/result.json) | Does Section 3.3 cite more papers than Section 3.1? Directly answer 'yes' or 'no'. | Yes | 1.0000 / yes | 0.0000 / no | 首词Yes与后面13对10的解释冲突；抽取最终选no。 |
| [#656](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0656/result.json) | How many departments/programmes are introduced in the "Research & Course Work Programmes@FASS" part (page 11-31)? | 17 | 1.0000 / 17 | 0.0000 / Not answerable | 所问11–31页区间未完全覆盖，且把缺少同名标题当作不能计数。 |
| [#668](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0668/result.json) | What is the central component of the figure at page 17? | Elastic Compute Service | 1.0000 / Elastic Compute Service | 0.0000 / A database icon with an arrow | 问题的印刷页17与物理页17混用，选中Data Migration图而非目标组件图。 |
| [#672](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0672/result.json) | How many cameras outside the China area does the figure at Page 19 has? Write the answer in integer format. | 3 | 1.0000 / 3 | 0.0000 / — | 页引用未解析到标注物理21，持续搜索后触顶。 |
| [#675](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0675/result.json) | Which cities does the China's largest cloud provider have teams? Write the answer in list format with alphabetical rder. | ['Dubai', 'Frankfurt', 'Hong Kong', 'London', 'New York', 'Paris', 'San Mateo', 'Seoul', 'Singapore', 'Sydney', 'Tokyo'] | 1.0000 / ['Dubai', 'Frankfurt', 'Hong Kong', 'London', 'New York', 'Paris', 'San Mateo', 'Seoul', 'Singapore', 'Sydney', 'Tokyo'] | 0.0000 / — | 正确城市数组被str类型检查拒绝。 |
| [#681](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0681/result.json) | What is the deadline month of the January intake ISEP students need to nominate supervisors and nominations of tac? Write the answer in list format, e.g., ["Jan","Feb"] | ['Jun', 'Aug'] | 1.0000 / ["Jun", "Aug"] | 0.0000 / — | 月份数组被类型拒绝；June/August与Jun/Aug还存在表达归一差异。 |
| [#685](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0685/result.json) | How many line plots are there in the report? | 8 | 1.0000 / 8 | 0.0000 / — | 全局线图排查耗尽轮数，未提交Reasoner。 |
| [#695](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0695/result.json) | How many % of Rep/Lean Rep people think cases have risen primally because of more testing and how many % of Rep/Lean Rep people think the federal government should be primarily responsible for COVID-19 control policy? | [62, 30] | 1.0000 / [62, 30] | 0.0000 / — | 没有读标注5/6，持续查询至预算耗尽。 |
| [#707](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0707/result.json) | What does a point mutation of the codon TTT or thymine-thymine define? | phenylalanine | 1.0000 / phenylalanine | 0.0000 / missense mutation | 目标页已提交，却把所问氨基酸答成突变类型。 |
| [#716](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0716/result.json) | What agency component costs the most of fund in the SRM Table? | Software Development | 1.0000 / Software Development | 0.0000 / Identification and Authentication | 多页SRM表已提供，费用最大项的列/分组比较错误。 |
| [#739](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0739/result.json) | What are the incident and report dates? Save your answer as a list, with each data formatted as 'YYYY-MM-DD'. | ['2015-03-24', '2015-03-25'] | 1.0000 / ['2015-03-24', '2015-03-25'] | 0.0000 / — | 日期数组被类型拒绝，数组中的报告日期本身也与标准不同。 |
| [#747](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0747/result.json) | How many figures illustrated the structure of fuel injectors? | 3 | 1.0000 / 3 | 0.0000 / — | 全局图示排查读了很多页但不提交Reasoner，预算耗尽。 |
| [#761](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0761/result.json) | How many strategies outperform the random selection baseline on CIFAR-10-LT? | 4 | 1.0000 / 4 | 0.0000 / 6 | 正确Table1实际在已提交的物理5；却依据另一Table7得出6，非简单gold未命中。 |
| [#767](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0767/result.json) | On the dataset map of PathMNIST, does the map by ground truth or pseudo-labels have a larger confidence variance? | ground truth | 1.0000 / ground truth | 0.0000 / pseudo-labels | 同图中ground truth/pseudo-label及置信度分散度关系判反。 |
| [#769](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0769/result.json) | Which figures include line plots in the paper? | ['Figure 5', 'Figure 6'] | 1.0000 / ['Figure 5', 'Figure 6'] | 0.0000 / — | 已读目标线图页5/6，却始终不向Reasoner提交，触顶。 |
| [#782](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0782/result.json) | What restaurants/teahouses in Qingyang District are recommended? Answer in English. | ['Chen Mapo Doufu', 'Yu Zhi Lan', 'Wenshu Monastery Teahouse'] | 1.0000 / ['Chen Mapo Doufu', 'Yu Zhi Lan', 'Wenshu Monastery Teahouse'] | 0.0000 / ["Chen Mapo Doufu", "Yu Zhi Lan", "Wenshu Monastery Teahouse", "Wenshu Monastery"] | 未读标注37/38，名单中多列Wenshu Monastery，列表长度不符。 |
| [#795](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0795/result.json) | How many images are shown in the section of trading capabilities? | 6 | 1.0000 / 6 | 0.0000 / 5 | 目标页齐全，漏计一张交易功能图片。 |
| [#796](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0796/result.json) | What is the button to click for up-to-the-minute portfolio balances? | Live Balances | 1.0000 / Live Balances | 0.0000 / — | 11次search+4次read，未读标注6，也没有Reasoner请求。 |
| [#804](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0804/result.json) | In the survey conducted May 1-15, 2017, what percentage of U.S. adults says the number of people killed or injured in traffic accidents will not decrease if driverless vehicles become widespread? | 61% | 1.0000 / 61 | 0.0000 / 31 | 已提供对应调查页，will not decrease与其他回答类别的百分比混淆。 |
| [#811](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0811/result.json) | Among all the emotions that social media makes users feel, what was the 4th most popular emotion? | Inspired | 1.0000 / Inspired | 0.0000 / Connected | 情绪排序/图表读数错误，第四名答Connected而非Inspired。 |
| [#813](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0813/result.json) | How many charts are shown in the first 10 pages of the document? | 5 | 1.0000 / 5 | 0.0000 / 4 | 前十页已覆盖，但漏计一个图表。 |
| [#818](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0818/result.json) | How many U.S. adults express their opinions on if social media provides an accurate picture of how society feels about important issues? | 4548 | 1.0000 / 4548 | 0.0000 / 74 | 只提交意见图页20，漏调查样本量页28，答成74%而非4548人。 |
| [#821](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0821/result.json) | how many tables are included in the document? | 3 | 1.0000 / 3 | 0.0000 / — | 已读全部表格标注页11/13/16，却不提交Reasoner，触顶。 |
| [#836](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0836/result.json) | What is the combined percentage of Album Sales % and Song Sales % for the Country genre? | 23 | 1.0000 / 23 | 0.0000 / 17 | 目标图已提交，Country的两项比例读数或求和错误。 |
| [#856](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0856/result.json) | How many all vendors have big data revenues over $250M? | 5 | 1.0000 / 5 | 0.0000 / 6 | 图表阈值/坐标读错，超过250M的公司计6而非5。 |
| [#858](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0858/result.json) | List the different senses in the ascending order of their bandwidth according to Page 20 | ['Taste', 'Hearing', 'Smell', 'Touch', 'Sight'] | 1.0000 / ['Taste', 'Smell', 'Hearing', 'Touch', 'Sight'] | 0.0000 / ["Taste", "Smell and Hearing", "Touch", "Sight"] | 将Smell和Hearing合成一项，五元素列表变四项。 |
| [#866](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0866/result.json) | According to this paper, among nice different datasets in Fig 4, which one can reflect the most breadth of knowledge? | "MMMU" | 1.0000 / MMMU | 0.0000 / ScienceQA | 未读目标物理3，读4/5/6后答ScienceQA而非MMMU。 |
| [#900](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0900/result.json) | How steps are needed to pair and connecting the headset with a computer (Mac)? | 4 | 1.0000 / 4 | 0.0000 / ['Enter pairing mode on the headset (turn on for first-time pairing, or press and hold the power button for about 7 seconds for subsequent p… | 问题问步骤数，输出步骤列表；且混入唤醒电脑等额外操作，不能只靠数列表项修复。 |
| [#915](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0915/result.json) | How many employees dose PWC have including Assurance, Consulting, Deals and Tax & Legal Services? | 5100 | 1.0000 / 5100 | 0.0000 / — | 部门页覆盖缺少12，最后只提交14并反复缺失，预算耗尽。 |
| [#921](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0921/result.json) | What percentage of Republicans strongly favor automatically registering all eligible citizens to vote? | 14% | 1.0000 / 14 | 0.0000 / 38 | 未读标注8，对strongly favor和其他群体/强度的比例取错。 |
| [#924](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0924/result.json) | How much did the proportion of the favorable views of China among the American public decresed from 2005 to 2020? | 21% | 1.0000 / 21 | 0.0000 / 30 | 原始值43和22均正确，却算出下降30。 |
| [#931](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0931/result.json) | How many lateral bar graphs are present in the document? | 10 | 1.0000 / 10 | 0.0000 / 12 | 全局横向柱图计数口径/去重错误，12而非10。 |
| [#935](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0935/result.json) | How many times does "HIT" appear in slide? | 3 | 1.0000 / 3 | 0.0000 / 4 | 目标页已提交，但HIT出现次数计4而非3。 |
| [#938](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0938/result.json) | WHAT IS ITC LIMITED LOGO COLOR? | Blue | 1.0000 / Blue | 0.0000 / Blue and white | Blue and white与标准Blue的颜色范围不同，且gold含0，页覆盖不作结论。 |
| [#942](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0942/result.json) | Among all in-office board members, how many other chairmanships the person who attended the least board meetings hold on the boards of other companies? | 4 | 1.0000 / 4 | 0.0000 / 0 | 已提交标注页，最少出席者与其其他董事长任职数量的关联错误。 |
| [#944](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0944/result.json) | What is total committee strengh of audit committee meetings and compensation committee meetings in 21st July, 2006? | 8 | 1.0000 / 8 | 0.0000 / [3, 5] | 取得两个数量3和5，却未求和，pred仍为[3,5]，Int评分零分。 |
| [#961](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0961/result.json) | How many different twitter accounts are shown in slide? | 2 | 1.0000 / 2 | 0.0000 / 1 | 已提交标注页，跨页账户枚举/去重漏项，1而非2。 |
| [#964](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0964/result.json) | What stages does high level lifecycle have more than basic lifecycle? | ['concept', 'production'] | 1.0000 / ['Concept', 'Production'] | 0.0000 / ["Concept", "Inception"] | 两组集合列对，差集却选错，Concept/Inception而非Concept/Production。 |
| [#965](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0965/result.json) | The author mainly conducted experiments on which two tasks? | ['event relation extraction', 'deductive reasoning'] | 1.0000 / ['Event Relation Extraction (ERE)', 'deductive reasoning'] | 0.0000 / Event Relation Extraction (ERE) and deductive reasoning tasks | 原文含两项正确任务，但抽取为单句，List元素数不符。 |
| [#969](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0969/result.json) | What subskill does we need to collect the available data? | semantic parsing | 1.0000 / Semantic Parsing | 0.0000 / semantic parsing to compile the available data | semantic parsing后保留长用途说明，短答案表达与ANLS不兼容。 |
| [#976](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0976/result.json) | How much does the Anchor Re-weighting method improve the performance of vanilla ICL on average? | 16.74 | 1.0000 / 16.74 | 0.0000 / 18.74 | 页已提供，比较均值所用68.64/49.90与目标提升口径不符。 |
| [#982](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0982/result.json) | When a student cannot be found, who should we contact? | Campus Security | 1.0000 / Campus Security | 0.0000 / Campus Security at (65) 6790 5200 or approach your Hall Fellow if you live in a Hall of Residence. | 原文含Campus Security，但抽取包含电话和其他联系人，长串ANLS归零。 |
| [#995](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0995/result.json) | How many weeks do students need to reach work experience milestone and get 8 units? | 16 | 1.0000 / 16 | 0.0000 / 8 | 未读标注10，取了另一实习选项的8周而非16周。 |
| [#1016](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1016/result.json) | How much shares did COSTCO repurchase in FY2021? | 1358000 | 1.0000 / 1358000 | 0.0000 / 1,358,000 | 正确1,358,000保留千分位，Float解析失败。 |
| [#1031](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1031/result.json) | what profit did Memberships contribute to in Domestic Streaming Segment in FY2015? Answer in thousands. | 1375500.00 | 1.0000 / 1375500 | 0.0000 / 1,375,500 | 正确1,375,500保留千分位，Float解析失败。 |
| [#1032](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1032/result.json) | what is the FY2021 inventory turnover ratio for Nike?Round your answer to two decimal places. | 3.46 | 1.0000 / 3.46 | 0.0000 / 1.81 | 标注财务页已提交，但存货周转率的取数/公式口径错误。 |
| [#1034](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1034/result.json) | What is payables turnover in FY2021 for Nike? Round your answer to two decimal places. | 9.67 | 1.0000 / 9.67 | 0.0000 / — | 同一页集6次请求，重复insufficient/索要公式，最后触顶。 |
| [#1048](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1048/result.json) | what is Amazon's FY2017 cash balance to total liability ratio? round your answer to three decimal | 0.198 | 1.0000 / 0.198 | 0.0000 / 0.355 | 只读39缺49，财务比率分子分母/口径不符。 |
| [#1064](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1064/result.json) | what is After-tax Return on Average Equity for the fiscal year ending January 28, 2023? round your answer to three decimal places | 0.488 | 1.0000 / 0.488 | 0.0000 / — | 未读标注38/39，财务取数失败后触顶。 |
| [#1066](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1066/result.json) | what is capitalization ratio for Best Buy for the fiscal year ending January 28, 2023? Answer in percentage term, round to one decimal places. | 29.6% | 1.0000 / 29.6 | 0.0000 / 10.1 | 财务目标页已提交，但capitalization ratio口径错误，10.1而非29.6。 |
| [#1080](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1080/result.json) | what is the net fair value of derivative instruments for 3M in FY2018? Answer in million. | 93 | 1.0000 / 93 | 0.0000 / 108 | 列出负债22+31却算38，造成净额108而非93。 |

## 10. 完整清单：Agent满分、e2e零分（64题）

以下是逐题输出比较和第一层诊断；页覆盖只作线索，具体因果以原文/轨迹和必要消融为准。长文本用省略号截短，完整值见CSV/JSON。

| 编号 | 问题 | 标准答案 | e2e得分 / pred | Agent得分 / pred | 诊断 |
| ---: | --- | --- | --- | --- | --- |
| [#12](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0012/result.json) | How many charts and tables in this report are sourced from Annual totals of Pew Research Center survey data? | 7 | 0.0000 / 5 | 1.0000 / 7 | 最终7得分，但分项加总为11；解释不一致，不能直接当优质成功轨迹。 |
| [#26](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0026/result.json) | How many exterior photos of organisations does this report provide? | 10 | 0.0000 / 7 | 1.0000 / 10 | Agent按多页照片枚举10处，e2e仅列7处；局部检索/覆盖更充分的候选优势。 |
| [#58](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0058/result.json) | As of Q3 2015, are there more active Instagram users in Vietnam, or are there more fans for Appota's fan page? Directly answer 'Instagram' or 'Appota'. | Appota | 0.0000 / Not answerable | 1.0000 / Appota | 两边原文均为Appota，e2e抽取却变Not answerable；纯抽取链路分差。 |
| [#64](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0064/result.json) | How many tables include "F1" as a metric? | 3 | 0.0000 / 4 | 1.0000 / 3 | e2e把Significance当F1额外计入，Agent聚焦含F1的表得3。 |
| [#66](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0066/result.json) | For dataset construction, which step takes the most word to describe than the others. | Evolutionary Question Generation | 0.0000 / Not answerable | 1.0000 / Evolutionary Question Generation | e2e因无明示字数而拒答，Agent比较描述区间答Evolutionary Question Generation。 |
| [#103](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0103/result.json) | Which colleges/schools/institutes from NTU use year 3 students as cover page student model? | ['College of Engineering', 'Nanyang Business School', 'National Institute of Education'] | 0.0000 / ['Nanyang Business School (NBS)', 'National Institute of Education (NIE)'] | 1.0000 / ["College of Engineering", "Nanyang Business School (NBS)", "National Institute of Education (NIE)"] | 聚焦三个封面学生页，补出e2e漏掉的College of Engineering。 |
| [#106](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0106/result.json) | How many goblets appear in the figure showing Skyskraoeren? | 7 | 0.0000 / 6 | 1.0000 / 7 | 目标餐桌页+可用crop辅助数出7杯，e2e6杯；局部视觉优势候选。 |
| [#109](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0109/result.json) | Which transport is included in "Norway in a Nutshell" but not "Sognefjord in a Nutshell" | bus | 0.0000 / None | 1.0000 / Bus | 同一旅游路线页辨出Bus差项，e2e误判两条路线交通完全相同。 |
| [#137](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0137/result.json) | Which individual lead this program? | Allison Crimmins | 0.0000 / Not answerable | 1.0000 / Allison Crimmins | 定向页集找到Allison Crimmins，e2e对program指代过于保守而拒答。 |
| [#174](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0174/result.json) | Which country had the highest percentage of respondents for whom traditional values mean a lot? | Oman | 0.0000 / 60 | 1.0000 / Oman | e2e原文已经包含Oman，但抽取成60；不是纯取证优势。 |
| [#178](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0178/result.json) | According to this survey, which country’s youth show the greatest concern about unemployment problem? | Egypt | 0.0000 / Algeria | 1.0000 / Egypt | 同一调查图中Agent选Egypt，e2e的颜色/国家比较选Algeria。 |
| [#190](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0190/result.json) | Tell me the prompt template used for label 'Contact.Meet' in filter-then-rerank method of this work. | The word {evt} triggers a MEET event: two or more entities come together at a single location and interact with one another face-to-face. | 0.0000 / Not answerable | 1.0000 / 'The word {evt} triggers a MEET event: two or more entities come together at a single location and interact with one another face-to-face.' | 集中到模板表页28定位Contact.Meet，e2e漏查该行后拒答。 |
| [#237](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0237/result.json) | What date is mentioned on the second page? Format the date as YYYY-MM | 2009-07 | 0.0000 / Not answerable | 1.0000 / 2009-07 | Agent定位日期并规范为2009-07；e2e直接拒答，页引用仍需核对。 |
| [#239](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0239/result.json) | How many square miles did the Hamilton country covers on year 1882? Return me a rounded integer. | 538 | 0.0000 / Not answerable | 1.0000 / 538 | e2e读到538却因1882的条件拒答；Agent按文档面积给出538，属于问题解释差异。 |
| [#243](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0243/result.json) | What is the residential capacity of Staten Island from 2003 to 2007? Give me an integer. | 435000000 | 0.0000 / 440980000 | 1.0000 / 435000000 | Agent取表中住宅容量435000000，e2e另加变化量，问题时间口径不同。 |
| [#299](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0299/result.json) | How many reasoning steps are involved in the figure 1 in the paper? | 5 | 0.0000 / 3 | 1.0000 / 5 | Agent把目标图分解为5个步骤，e2e只计3个推理节点。 |
| [#315](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0315/result.json) | How many figures in page 13 contains the left arrows? Write down the number in integer format. | 2 | 0.0000 / 4 | 1.0000 / 2 | 目标图中箭头/图数区分更准确，Agent2而e2e4。 |
| [#317](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0317/result.json) | In the adjusting the upper basket step, how many up arrows are contained in the figures? Write down the answers in integer format. | 1 | 0.0000 / 2 | 1.0000 / 1 | Agent分清上/下箭头方向，1而非e2e的2。 |
| [#318](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0318/result.json) | To built-in install for the integrated diswasher model, how many mm should the diswasher leave between the cabinet bottom and floor? | 100 | 0.0000 / 50 | 1.0000 / 100 | 聚焦安装尺寸图，100mm而非e2e的50mm。 |
| [#335](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0335/result.json) | Which programme by coursework with disciplinary content allows to have the maximum of 3 years full-time duration? Write down the answer with alphabetical order. | ['MA (Humanities Education)', 'MSc (Exercise & Sport Studies)', 'MSc (Life Science)', 'MSc (Mathematics for Education)'] | 0.0000 / Not answerable | 1.0000 / ["MA (Humanities Education)", "MSc (Exercise & Sport Studies)", "MSc (Life Sciences)", "MSc (Mathematics for Educators)"] | 对目标学制表的类别/列匹配不同，Agent补齐四项，e2e按research列拒答。 |
| [#345](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0345/result.json) | What is the percentage difference between the proportion of people who are very confident in Trump's ability to make wise decisions about immigration policy and those who are not confident at all? | 16% | 0.0000 / 18 | 1.0000 / 16 | 结果16正确，但首句13与后续45−29=16矛盾。 |
| [#353](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0353/result.json) | How many EU people believe that they will have more influence in world affairs after the coronavirus outbreak compared to before the outbreak? | 19% | 0.0000 / Not answerable | 1.0000 / 19 | e2e读到19%却将EU people解释为EU受访人群而拒答，Agent采用指标对象解释。 |
| [#418](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0418/result.json) | How many people in India were using a debit card when the e-commerce sales reached 11 billion? | 399 million | 0.0000 / 399 | 1.0000 / 399 million | 两边原文均正确399 million，e2e抽取丢million而零分。 |
| [#433](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0433/result.json) | Which animal did the detector ultimately recognize the person as? | Dog | 0.0000 / ['cat', 'dog'] | 1.0000 / dog | Agent选最终识别结果dog，e2e把前后两次结果cat/dog合并。 |
| [#476](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0476/result.json) | How many steps are there for data exchange via USB? | 9 | 0.0000 / Not answerable | 1.0000 / 9 | Agent把USB菜单动作逐项计为9；e2e要求明确步骤说明而拒答，属于计数口径差异。 |
| [#492](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0492/result.json) | How many triangles appear on the eighth side? | 4 | 0.0000 / Not answerable | 1.0000 / 4 | Agent将eighth side解释为目标页并数4个三角形，e2e将side理解为几何边而拒答。 |
| [#504](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0504/result.json) | What proportion of the Spanish dominant Latinos express a negative impression of socialism? | 46% | 0.0000 / 54 | 1.0000 / 46% | 同图正/负印象读数判反：Agent46，e2e54。 |
| [#517](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0517/result.json) | Within the geographic market segments defined by the College Board Search platform, which of the two designated areas, CA19 or Covina and West Covina, encompasses a larger geographic region? | Covina and West Covina | 0.0000 / Not answerable | 1.0000 / Covina and West Covina | Agent用地图形状比较面积，e2e要求文字面积数值而拒答。 |
| [#531](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0531/result.json) | What is the average session duration for desktop? | 00:04:14 | 0.0000 / 254 | 1.0000 / 00:04:14 | 两边原文都是00:04:14，e2e抽取转为254秒，表示不同而零分。 |
| [#539](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0539/result.json) | In Figure 122, what's the colour of OSCARb in the prompt? | Blue | 0.0000 / Not answerable | 1.0000 / blue | e2e看到blue却对in the prompt作严格文字解释而拒答；Agent按图示颜色回答。 |
| [#540](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0540/result.json) | In Figure 113, what's the name of the music app shown in the computer screen? | YouTube Music | 0.0000 / Not answerable | 1.0000 / YouTube Music | 定位物理142并补一张截图crop，识别小字YouTube Music；e2e称图标不清。 |
| [#542](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0542/result.json) | In Figure 107, what's the battery percentage shown in the screenshot? | 76% | 0.0000 / 70 | 1.0000 / 76 | 局部截图电量读76%，e2e读70%。 |
| [#568](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0568/result.json) | What is the percentage difference between the sum of women who have received a STEM degree and are employed in their field, and the sum of men with the same status in the survey of Pew Research Center, January 2018? | 15% | 0.0000 / 21 | 1.0000 / 15 | 最终15正确，所列男女加数却不等于文中114/99；过程一致性不足。 |
| [#593](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0593/result.json) | Which officer verify and signed the complaint? | Special Agent Marc Silski | 0.0000 / Marc Silski | 1.0000 / Special Agent Marc Silski | 两边已识别Marc Silski，Agent保留Special Agent头衔，ANLS计分有利。 |
| [#598](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0598/result.json) | What is the sum of numbers that are written manually on the page 15 and page 16? Write the answer in the integer format. | 19 | 0.0000 / 24 | 1.0000 / 19 | 聚焦两页手写数字求和19，e2e读数/求和为24。 |
| [#607](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0607/result.json) | What is the sum of percentage of customers and employees said that Wi-Fi increases customer loyalty in hospitality in 2015? | 122 | 0.0000 / Not answerable | 1.0000 / 122 | 目标页36/37的61+61=122；e2e编号重复到截断。 |
| [#621](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0621/result.json) | How many website URLs are contained in the slides? | 3 | 0.0000 / 2 | 1.0000 / 3 | 最终3正确，但把两个.wmv文件名当website URLs计数；证据解释存疑。 |
| [#627](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0627/result.json) | For all the examples given in Figure 5 and Figure 6, how many of them Logic-LM predicted the correct answer? | 2 | 0.0000 / 5 | 1.0000 / 2 | Agent逐案例核对Logic-LM对错，2而非e2e误计的5。 |
| [#629](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0629/result.json) | How many prompt examples are given in the Appendix? | 15 | 0.0000 / 5 | 1.0000 / 15 | 按每数据集三类prompt聚合15，e2e只按五个数据集各计一个。 |
| [#648](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0648/result.json) | For all the papers that are listed in Table 1 and Table 2, how many of them are proposed in 2021? | 1 | 0.0000 / 2 | 1.0000 / 1 | 跨表筛年份得到1，e2e多纳入一篇2021条目；需保留文献口径核对。 |
| [#652](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0652/result.json) | Which of the following department does not provide a PhD programme, Department of Economics, Department of English Language and Literature, or Department of History? | Department of History | 0.0000 / Not answerable | 1.0000 / Department of History | Agent按给定三部门比较PhD项目，e2e判断三个都有而拒答。 |
| [#655](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0655/result.json) | For all the people that appear in the guidebook to speak for their department or programmes, how many of them are assistant professors? | 1 | 0.0000 / 2 | 1.0000 / 1 | 人物/职位归属范围解释不同，Agent1、e2e2；不能只据得分确认Agent覆盖更全。 |
| [#679](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0679/result.json) | How many areas do the course requiremnt have? | 4 | 0.0000 / 2 | 1.0000 / 4 | Agent整合相邻页得到四项，e2e仅看首表两项。 |
| [#752](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0752/result.json) | What task is on the top of the planning branch of the tree? | egocentric planning | 0.0000 / Indoor Event Planning | 1.0000 / Egocentric Planning | 局部树图的最上分支读取更准确，Egocentric Planning而非Indoor Event Planning。 |
| [#763](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0763/result.json) | What querying criteria yields the worst label diversity generally on all the datasets? | VAAL | 0.0000 / Not answerable | 1.0000 / VAAL | Agent对比VAAL多图指标，e2e要求明示最差结论而拒答。 |
| [#776](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0776/result.json) | At which step, the loss spikes when without QK-norm? | 150k | 0.0000 / 40000 | 1.0000 / 150k | 目标损失曲线读150k，e2e把步骤轴读成40k。 |
| [#777](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0777/result.json) | What is the first animal, other than humans, shown in this guidebook? | panda | 0.0000 / A giant panda | 1.0000 / panda | 同为panda语义，e2e多写A giant导致ANLS零分，不是识别能力差异。 |
| [#803](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0803/result.json) | What are the five steps of ValueEdge Insights? | ['Plan', 'Build', 'Test', 'Deliver', 'Run'] | 0.0000 / ['Plan, build, and deliver digital product value streams.', 'Discover, visualize, and manage flow—from strategy to delivery.', 'Accelerate b… | 1.0000 / ["Plan", "Build", "Test", "Deliver", "Run"] | 定位真实五阶段示意图，e2e答成平台介绍的五条价值陈述。 |
| [#825](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0825/result.json) | How many slides include more than one figure? Here don't consider the charts. | 1 | 0.0000 / 0 | 1.0000 / 1 | Agent扫完20页并聚焦多图页，判1；e2e全局扫描漏该页。 |
| [#826](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0826/result.json) | How many lines are highlighted in yellow on the first page? Give me an integer. | 2 | 0.0000 / 3 | 1.0000 / 2 | 同一页黄色高亮数2，e2e3。 |
| [#831](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0831/result.json) | How many critical thinking case studies are included in all the assignments? | 3 | 0.0000 / 4 | 1.0000 / 3 | Agent区分具名case study与泛指活动，e2e多计一个。 |
| [#832](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0832/result.json) | how many LEARNING OUTCOMES should be ANSWERed in UNIT 8? | 10 | 0.0000 / 9 | 1.0000 / 10 | 枚举10项让抽取器得到正确10，但原文首句仍说7。 |
| [#869](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0869/result.json) | List all the error types mentioned in the case studies in Pages 95-100 | ['Reasoning Error', 'Perceptual Error', 'Lack of Knowledge'] | 0.0000 / ['Perceptual Error', 'Lack of Knowledge'] | 1.0000 / ["Reasoning Error", "Perceptual Error", "Lack of Knowledge"] | 定向95–100区域补出Reasoning Error，e2e漏这一类。 |
| [#927](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0927/result.json) | Assume that after July 2020, the percentage of Republicans who believe "China is an enemy of the U.S." increased by 10 percentage points, while the percentage of Democrats remained the same. What would be the difference in the per… | 29% | 0.0000 / Not answerable | 1.0000 / 29 | 最终48−19=29正确，但首句19与之矛盾；gold页也可能偏移。 |
| [#928](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0928/result.json) | which age group experienced the greatest change in the percentage holding an "unfavorable" opinion of China between 2005 and 2010? | 50 and older | 0.0000 / 30-49 | 1.0000 / 50 and older | 局部年龄曲线比较为50 and older，e2e取错30–49。 |
| [#929](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0929/result.json) | What is the title of the slide? | Medical Informatics and HIT Systems | 0.0000 / Not answerable | 1.0000 / Medical Informatics and HIT Systems | Agent聚焦封面标题；e2e列出多页标题但未给单一答案，最后弃答。 |
| [#947](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0947/result.json) | What is the date of the document? Format your answer as YYYY-MM-DD. | 2022-01-05 | 0.0000 / 2022-05-01 | 1.0000 / 2022-01-05 | 同一个01/05/2022日期的月日解释不同，Agent2022-01-05符合标准。 |
| [#978](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0978/result.json) | What is the most confusing category of Abbreviation? | Description | 0.0000 / Entity | 1.0000 / Description | 混淆矩阵的条件/方向解释不同，AgentDescription，e2eEntity。 |
| [#988](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0988/result.json) | Where did Terry Tan study the foundation course before he progressed to NTU? | NTIC | 0.0000 / Nottingham Trent International College | 1.0000 / NTIC (Nottingham Trent International College) | NTIC与Nottingham Trent International College为同一机构；简称/全称表达导致评分差异。 |
| [#996](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0996/result.json) | How many different iPad solutions are shown in the document? | 4 | 0.0000 / 5 | 1.0000 / 4 | 不同iPad方案的类别/去重边界不同，Agent4、e2e5。 |
| [#1028](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1028/result.json) | what method did netflix use to pay the dividend to shareholders in FY2015. | seven-for-one stock split | 0.0000 / Not answerable | 1.0000 / seven-for-one stock split | Agent找到stock dividend对应seven-for-one split，e2e只关注无cash dividend而拒答。 |
| [#1045](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1045/result.json) | what is Amazon's FY2017 return on equity? round your answer to three decimal | 0.109 | 0.0000 / 0.129 | 1.0000 / 0.109 | ROE分母选择差异，Agent0.109、e2e平均权益口径0.129；应审计题目公式口径。 |
| [#1059](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1059/result.json) | What is Cash Flow to Total Debt Ratio for ADBE In FY2015? | 0.77 | 0.0000 / Not answerable | 1.0000 / 0.77 | 集中现金流与债务算0.77；e2e已找原始量却因无直接比率而拒答。 |
| [#1071](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1071/result.json) | what proportion does Best Buy have female employees in new, salaried field positions for the fiscal year ending January 28, 2023? | 25% | 0.0000 / Not answerable | 1.0000 / 25 | e2e看到25%却因财政年度结束日条件过度保守拒答；Agent直接给25%。 |

## 11. 其他得分不同案例：涉及部分分（43题）

以下是逐题输出比较和第一层诊断；页覆盖只作线索，具体因果以原文/轨迹和必要消融为准。长文本用省略号截短，完整值见CSV/JSON。

| 编号 | 问题 | 标准答案 | e2e得分 / pred | Agent得分 / pred | 诊断 |
| ---: | --- | --- | --- | --- | --- |
| [#0](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0000/result.json) | According to the report, how do 5% of the Latinos see economic upward mobility for their children? | Less well-off | 0.9231 / "Less well off" | 0.0000 / 5% of Latinos expect their children to be less well off financially than they themselves are now. | e2e得0.9231而非满分，此题不计入129。Agent回答到less well off，抽取却保留整句。ANLS测字符串距离，不是对原推理的语义判定。 |
| [#10](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0010/result.json) | According to this report, which group is greater in population in 2018? Republican who are hispanic, or no leans who are male? | no leans who are male | 1.0000 / no leans who are male | 0.9524 / No lean who are male | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#15](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0015/result.json) | What range does red color represents in approximate distance from the Mississippi River in the chart that tracks the West NIle Virus in Europe? | 0-375 miles | 0.0000 / Not answerable | 0.8462 / 0 - 375 miles | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#42](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0042/result.json) | According to the chart "Levels of Analytics", what are the four business analystics activities? | ['OPTIMISATION', 'PREDICTIVE MODELING', 'FORECASTING', 'STATISTICAL ANALYSIS'] | 0.0000 / ['Forecasting', 'Statistical Analysis', 'Alerts', 'Query Drilldown (OLAP)'] | 0.9500 / ["Statistical Analysis", "Forecasting", "Predictive Modelling", "Optimisation"] | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#46](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0046/result.json) | Which Youtube does the slides use to show the consequce of blindly following data? | Girl crash into lake following bad GPS direction | 0.9600 / "Girls Crash into Lake following Bad GPS directions" | 0.0000 / CrushingBastards | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#49](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0049/result.json) | According to the slides, what are the things that a data-driven person has? | ['a goals first approach', 'an objective, inquisitive culture', 'a visible, clear articulated strategy', 'strong data leadership', 'a strong… | 0.9231 / ['A strong testing culture', 'An open, sharing culture', 'A self service culture', 'A broad data literacy', 'A goals first approach', 'An ob… | 0.0000 / ["a strong testing culture", "a broad data literacy", "a goals first approach", "a visible clearly-articulated strategy", "strong data leade… | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#71](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0071/result.json) | Based on "PSEUDOCODE FOR RETRIEVAL METHODS", which algorithm has more number of lines? (Give the algorithm name) | Collapsed Tree Algorithm | 1.0000 / Collapsed Tree Algorithm | 0.6667 / Algorithm 2 Collapsed Tree Algorithm | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#97](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0097/result.json) | What will happen when you press and hold the down button? | Wake up the voice assistant. | 0.9643 / Wake up the voice assistant | 0.0000 / The voice assistant will wake up. | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#99](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0099/result.json) | Which groups of applicants have the lastest end of application period according to this brochure? | ["Singapore-Cambridge GCE 'A' Level", 'International Baccalaureate (IB) Diploma'] | 0.6889 / ['Singapore-Cambridge GCE "A" Level certificate', 'International Baccalaureate (IB) Diploma'] | 0.0000 / ["Diploma Awarded by a Polytechnic or equivalent institution in Singapore", "International Baccalaureate (IB) Diploma"] | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#127](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0127/result.json) | What is the title of the of the Figure 2? | Diagram of Breccia Gashes with top denuded off but bottom shown in cliff | 1.0000 / Diagram of Breccia Gashes with top denuded off but bottom shown in cliff | 0.9863 / Diagram of Breccia Gashes with top denuded off but bottom shown in cliff. | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#140](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0140/result.json) | What are the last 4 chapters listed in the table of contents? | ['Climate Impacts on Water-Related Illness', 'Food Safety,Nutrition,and Distribution', 'Mental Health and Well-Being', 'Populations of Conce… | 0.0000 / ['Chapter 7: Food Safety, Nutrition, and Distribution', 'Chapter 8: Mental Health and Well-Being', 'Chapter 9: Populations of Concern', 'App… | 0.8400 / ["6. Climate Impacts on Water-Related Illness", "7. Food Safety, Nutrition, and Distribution", "8. Mental Health and Well-Being", "9. Popula… | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#169](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0169/result.json) | Enumerate the media sources surveyed in this report. | ['Radio', 'Newspaper', 'Televison', 'Internet'] | 0.0000 / ['Radio', 'Television', 'Newspaper', 'Internet', 'Mobile Phone', 'Friends and Family'] | 0.9000 / ["Radio", "Television", "Newspaper", "Internet"] | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#175](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0175/result.json) | What is the 8th (out of top10) findings listed in this presentation? | Arab youth are increasingly concerned about obesity and lifestyle diseases and do not believe that healthcare in their country is improving | 1.0000 / ARAB YOUTH ARE INCREASINGLY CONCERNED ABOUT OBESITY AND LIFESTYLE DISEASES AND DO NOT BELIEVE THAT HEALTHCARE IN THEIR COUNTRY IS IMPROVING | 0.9929 / Arab youth are increasingly concerned about obesity and lifestyle diseases and do not believe that healthcare in their country is improving. | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#188](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0188/result.json) | Repeat the instructions corresponding to the settings shown in red box of Figure 3 (left). | Identify the entities expressed by each sentence, and locate each entity to words in the sentence. The possible entity types are: [Type_1], … | 0.0000 / Not answerable | 0.9398 / Instruction1: Identify the entities expressed by each sentence, and locate each entity to words in the sentence. The possible entity types a… | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#204](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0204/result.json) | According to this guideline, what is on the left side of the MacBook Air's camera? | Ambient light sensor | 1.0000 / ambient light sensor | 0.8333 / The ambient light sensor | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#285](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0285/result.json) | In the figure that locates at the top of page 5, what is the final step? Please write down the answer in string format. | 4. The final prediction result with rationale. | 0.9130 / The final prediction result with rationale | 0.5625 / present the final prediction label with the supporting rationale | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#296](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0296/result.json) | What is the evaluation metric that has highest number for the method located at the bottom of the model structure figure across the three datasets? Please write down the answer with string format. | H@10 | 0.0000 / NDCG@3 | 0.7500 / N@10 | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#311](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0311/result.json) | In the table at page 28, which signal has the least frequency in the guidebook? | "3840 x 2160" at 30 Hz | 0.5238 / 3840 x 2160 | 0.9524 / 3840 x 2160 at 30 Hz | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#334](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0334/result.json) | Which program has the longest time period among all the leadership programmes? | Leaders in Education Programme | 1.0000 / Leaders in Education Programme | 0.8824 / The Leaders in Education Programme | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#402](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0402/result.json) | What are the three steps of binary fission of a prokaryotic cell? | ['Duplication of chromosome and separation of copies', 'Continued elongation of the cell and movement of copies', 'Division into two daughte… | 0.6322 / ['Duplication of the chromosome and separation of the copies', 'Continued elongation of the cell and movement of the chromosome copies to op… | 0.0000 / ["A single circular chromosome duplicates, and the copies begin to separate from each other", "The cell elongates, and the chromosomal copie… | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#404](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0404/result.json) | What animal is on the cover of each chapter? | leopard | 1.0000 / Leopard | 0.7778 / A leopard | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#412](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0412/result.json) | What is the ebay.in listing for shown in the image? | milk giving black young buffalo 20 liter per day | 0.6458 / milk giving black young buffalo | 1.0000 / milk giving black young buffalo 20 liter per day | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#426](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0426/result.json) | What types of animals appear in the PPT? | ['Dog', 'Cat', 'Bear'] | 1.0000 / ['cat', 'dog', 'bear'] | 0.7500 / ["cats", "dogs", "bear"] | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#438](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0438/result.json) | In the SnapNTell model architecture, what is the image used as an example input in the paper? | Eiffel Tower | 0.7500 / The Eiffel Tower | 1.0000 / Eiffel Tower | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#444](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0444/result.json) | In the pipeline diagram of the RAR model, which type of organism is used as the input case? | Butterfly | 1.0000 / Butterfly | 0.8182 / A butterfly | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#474](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0474/result.json) | In the picture on page 50, what time does the blue bar start? | 15:00 | 0.6000 / 21:00 | 1.0000 / 15:00 | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#485](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0485/result.json) | What models of cars are introduced on pages 10 and 11? | ['SE', 'SE Nightshade', 'XSE', 'XSE6', 'TRD', 'LE Hybrid'] | 0.6667 / ['SE', 'SE Nightshade', 'XSE', 'XSE V6', 'TRD', 'LE Hybrid'] | 0.0000 / ["XSE V6", "TRD", "LE Hybrid", "SE Hybrid", "XLE Hybrid", "XSE Hybrid"] | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#500](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0500/result.json) | What's the color of the Third of higher generation bar in the chart about Immigrant generations and U.S. Latinos? | Gray | 0.7500 / grey | 0.0000 / brown | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#506](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0506/result.json) | Which Hispanic origin groups with less than 60% holding the view that People NOT seeing racial discrimination where it really DOES exist? Return me a list please. | ['Cubans', 'Central Americans'] | 0.0000 / ['Mexican', 'Cuban', 'Central American'] | 0.8333 / ["Cuban", "Central American"] | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#526](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0526/result.json) | Which account tweets content tagged #WednesdayWisdom? | Target | 1.0000 / Target | 0.8571 / @Target | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#702](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0702/result.json) | The document represents which judges' opinions? | Scott Bradley. | 0.7500 / E. Scott Bradley | 0.5455 / Judge E. Scott Bradley | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#703](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0703/result.json) | Who is the defendant of this case? | DELAWARE STATE PUBLIC INTEGRITY COMMISSION. | 0.9767 / Delaware State Public Integrity Commission | 0.0000 / Diane Hanson | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#710](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0710/result.json) | Which figures depict mutation mechanisms with the double-chain DNA structure? | ['Figure 3', 'Figure 4', 'Figure 11'] | 0.7778 / ['Figure 4', 'Figure 5', 'Figure 6'] | 0.0000 / Figure 3 | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#743](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0743/result.json) | In which system the throttle valve is placed beneath the fuel injector? | Single Point Fuel Injection | 0.7778 / Multipoint Fuel Injection | 0.0000 / Not answerable | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#797](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0797/result.json) | Which type of chart is shown in the what-if analysis? | pie chart | 1.0000 / Pie chart | 0.9000 / Pie charts | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#823](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0823/result.json) | Which application software interfaces are displayed in the slides? | ['Microsoft Office OneNote', 'SOAPWare'] | 0.8276 / ['Microsoft Office OneNote 2003', 'SOAPware'] | 0.0000 / ["Microsoft Office OneNote", "Windows Journal", "Microsoft Educational Pack for Tablet PC", "Microsoft Experience Pack for Tablet PC", "Stic… | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#830](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0830/result.json) | What is the title of case study given under unit-8 in table? | Discrimination in the Workplace Continues | 0.6508 / Discrimination in the Workplace Continues—submit on Google Docs | 1.0000 / Discrimination in the Workplace Continues | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#873](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0873/result.json) | list the top-3 models in Figure 3 | ['Vicuna13b-v1.3', 'PaLM Bison', 'Vicuna13b-v1.1'] | 0.0000 / ['Llama-2 7b-chat', 'Llama-2 13b-chat', 'Llama-2 70b-chat'] | 0.8667 / ["Vicuna 33b-v1.3", "PaLM Bison", "Vicuna 13b-v1.1"] | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#945](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0945/result.json) | What company is a plaintiff? | LIBERTARIAN PARTY OF GEORGIA, INC | 0.0000 / "LIBERTARIAN PARTY OF GEORGIA, INC., a Georgia nonprofit corporation" | 0.8684 / The Libertarian Party of Georgia, Inc. | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#952](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0952/result.json) | In the "Who are you talking to" section, what tips does the author give us? | ['line them up', 'observe your clients', 'build a buyer persons'] | 0.9524 / ['Line them up', 'Observe your clients', 'Build a buyer persona'] | 0.0000 / ["Line them up by listing common characteristics of your best clients (hobbies, age, vacation preferences).", "Observe your clients by not o… | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#958](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_0958/result.json) | Cold is the catalyst for what reaction? | the chemical reaction involving CFCs. | 0.0000 / CFCs | 0.8649 / chemical reaction involving CFCs | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#1060](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1060/result.json) | what channel is the majority of ADBE product sold by in FY2015? | a software subscription model | 0.9310 / software subscription model | 0.0000 / — | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |
| [#1068](/media/data5/leizh/longdoc/longdoc/output/agent/a100_4gpu_direct_pages/samples/sample_1068/result.json) | how does Best Buy record Merchandise inventories? | the lower of cost or net realizable value | 0.9318 / at the lower of cost or net realizable value | 0.0000 / Best Buy records merchandise inventories at the lower of cost or net realizable value. The weighted-average method is used to determine cost… | 涉及部分分，详见双方原始答案；不自动解释为完全正确与完全错误。 |

## 12. 复现与附件

工作区附件目录：`output/analysis/e2e_vs_agent_answerable_20261109/`。

- `all_answerable_cases.csv/json`：859题完整对齐结果，包含完整原文、预测、页集和轨迹路径。
- `different_score_cases.csv`：236题得分差异清单。
- `trace_summaries.json`：上述236题的工具调用、实际Reasoner页集/图像数/crop数及响应摘要。
- `summary.json`：矩阵、源文件SHA256和计分复算校验。
- `collect_traces.py`：从原始日志重新收集摘要；`build_report.py`：检查数据并重建本报告附录。

```bash
cd /media/data5/leizh/longdoc/longdoc
PYTHONDONTWRITEBYTECODE=1 /media/data5/leizh/longdoc/.venv_grpo/bin/python -B \
  output/analysis/e2e_vs_agent_answerable_20261109/collect_traces.py
PYTHONDONTWRITEBYTECODE=1 /media/data5/leizh/longdoc/.venv_grpo/bin/python -B \
  output/analysis/e2e_vs_agent_answerable_20261109/build_report.py
```

复现不调用模型，不能代替修复后的真实回归实验。报告只添加分析产物，没有修改Agent/e2e的生产代码或实验原始预测。
