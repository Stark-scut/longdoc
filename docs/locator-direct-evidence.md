# Locator直接选页与精简读页

分支：feat/locator-direct-evidence。基于chore/repository-layout的ef2d427。
旧Agent源码、完整数据、旧运行快照不改动。

## 工具协议

```json
{"name":"read_page","arguments":{"pages":[3,8]}}
{"name":"call_reasoner","arguments":{"pages":[3,8]}}
```

read_page一次最多页数仍由read_page_max控制，默认3；只改变成功返回内容。
成功role=tool消息按页交替包含page_index文字和真实image_url，保留匹配tool_call_id。
没有OCR、crop、标题或完整页记录JSON。失败时仍返回可诊断错误。

Locator通过call_reasoner明确给出全部证据页，每次替换集合；后端仅验证页号与已读状态，
然后把指定页的整页图、OCR、crop交给Reasoner。没有额外模型选页。
证据版本仅在选择集合变化时增加。选择非法页不会改变原证据或产生Reasoner请求。

search的OCR摘要和初始文档大纲仍保留，这是定位信息；“无OCR”约束针对read_page返回。
Locator历史仍持久保留页面图像。Reasoner返回的回答/缺项可能引用证据内容，属于推理结果。

## 配置与结果隔离

移除selector角色、模型调用、代码文件和提示词。
schema提升为2，配置显式记录pipeline=locator-direct-evidence、locator_read_page_format=image-only。
此前Selector输出需要原分支代码续跑；新模式使用新output/agent目录。
原有预算、thinking开关、并发、快照、回放和评分口径保持。

本次改动同时包含取消Selector和减少Locator观察，真实评测差异不构成单因素消融结论。

## 根目录整理

19个旧兼容脚本/文档链接移至compat/；根目录不再留测试/训练/评测脚本。
实际代码分别在agent/、evaluations/、examples/、training/、tests/，文档在docs/。
旧包装器修正相对路径，兼容文档链接从compat/指向正式文件。
已有大数据目录与ZIP按要求保持原位；运行输出与缓存集中到output/。

## 验证重点

- Locator下一轮实际请求不含read_page的OCR/crop；每页恰好一张完整图。
- Reasoner仍获得选定页的OCR和crop。
- 第二次提交[8]不夹带前次[3]。
- 未读页、空列表、布尔/字符串页号、缺参、额外参数被拒绝且可纠正。
- 全链路无Selector请求、角色或工具。
- 协议1输出不允许按协议2续跑。
- 并发、中断续跑、回放与GT隔离继续受离线测试覆盖。


## 本次验收（2026-10-29）

- tests/test_agent.py：28项离线测试通过。
- tests/test_answer_extraction.py：4项离线测试通过。
- 新Agent与compat、分类脚本的Python/shell语法检查通过，compat文档链接有效。
- 旧Agent的20个程序文件哈希未变；完整数据和ZIP保持原位。
- 真实数据第522题的索引10页：Locator返回1张整页图和page_index=10；Reasoner同页返回17个图像项及2941字符的页面/OCR标注。
- 未启动真实GPU推理、未报告新的ACC/F1。两个流程变化同时发生，后续应使用独立输出目录做对照。


## 2026-10-29：五题全为premature_final的修复

真实运行direct_pages_smoke的5题只有Locator请求（共27次），全部在直接答题后结束。
关键答案与GT相符，但Reasoner和extractor调用均为0，所以原成绩按未完成计零。
这是结束路径问题，不能把0分解释为5道视觉题全部答错。原结果不改动。

修复默认required工具调用、加强路由提示，并加入有上限的控制器协议纠正。
纠正仍由Locator真实返回call_reasoner及页号，Reasoner独立看证据；轮数、时间和成本不豁免。
新增加6项覆盖该真实遗漏的测试，34项Agent测试与4项判分测试通过。
修复后的真实GPU评测尚未运行，使用output/agent/direct_pages_smoke_v2等新目录。
