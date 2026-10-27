# 用 DeepSeek 重提取 MMLongBench 短答案

输入默认为原 qwen35_1024_2replicas_bf16/merged.json，共 1082 题，1019 题有原始 response、63 题没有回答。仅重提取已有回答，不重新生成文档 QA。GT 不发给 API，留在本地沿用原官方 eval_score.py 评分；无原始回答的 63 题继续计零，全部 1082 题保持在分母中。

~~~bash
conda activate qwen35
python /media/data1/leizh/longdoc/reextract_mmlongbench_deepseek.py --concurrency 16
~~~

若环境中没有 DEEPSEEK_API_KEY，交互终端会隐藏输入密钥；密钥不存进代码、文件、命令参数或日志。也可以通过已有安全方式将密钥设置到 DEEPSEEK_API_KEY。默认模型 deepseek-flash，thinking enabled，reasoning_effort high，与用户提供的 API 调用一致。

默认输出：output/mmlongbench/qwen35_1024_2replicas_bf16/deepseek_flash_reextract/
- merged.summary.json：新旧 ACC/F1、分数差、覆盖率、失败数、改善/退步题数、API usage。
- merged.json：保留原始 QA 回答与 original_evaluation，替换为新提取答案及评分。
- changes.json：预测或分数发生变化的逐题对照（含原始回答、GT、新旧短答案及分数）。
- samples/sample_0058/result.json：每题的持久化结果。
- samples/sample_0058/api_0001.json：每次 API 响应及 usage，不含请求鉴权头。
- source_merged.json、official_eval_score.py、extraction_prompt.md、system_prompt.txt、config.json：冻结来源与提取/评分配置。

先预检已知错误的第 58 题，之后并发处理其他题。原来提取成功的题也重新提取；成功检查点续跑跳过；失败题在下次启动重新尝试。接口认证、余额、模型或参数错误会停止任务，避免批量重复失败。每题最多重试 3 次；输出截断不评分，后续重试扩大输出预算。中断时已完成结果保留，正在进行的请求可能已被服务端计费，日志标记未知状态，不虚构 token 成本。

再次执行相同命令可续跑。提取配置或输入变化时需要新的输出目录。--limit N / --indices 58,0 可限制本次请求范围，汇总仍采用全量分母；--merge-only 仅汇总检查点，不请求 API、不需要密钥。--dry-run 只查看配置。新旧分数在重提取全部完成前不可作为最终结论。

官方 ACC 会计入部分字符串匹配分数；F1 为 MMLongBench 的答案可回答性 F1，不是常见的 token overlap F1。保留原官方提取示例，另用系统指令明确只能提取原模型的回答，不能自行纠正答案或因为没有 PDF 而否定明确的短回答。消息统一为 system + user，原 QA 回答作为数据放入 user，不伪造 assistant 历史。

离线测试（无真实 API 请求）：
~~~bash
python /media/data1/leizh/longdoc/tests_reextract_mmlongbench_deepseek.py
~~~

API 文档：https://api-docs.deepseek.com/api/create-chat-completion/
