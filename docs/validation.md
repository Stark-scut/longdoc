# 初始整理版本验证记录（ef2d427）

日期：2026-10-28。分支：chore/repository-layout。

最终检查结果：

| 检查 | 结果 |
| --- | --- |
| python tests/test_agent.py | 21项通过 |
| python tests/test_answer_extraction.py | 4项通过 |
| python citeagent_v2/full_eval/tests.py | 14项通过 |
| Python语法与shell语法 | 通过 |
| 旧核心/冻结Agent源码哈希 | 20个文件未变 |
| 迁移后的固定基线manifest | 52项校验通过 |
| 完整数据和ZIP位置 | 未变 |
| 新包脱离旧仓库运行 | 通过，复制agent包到临时目录，用合成数据和模拟HTTP服务完成评测 |
| 真实数据读取预检 | 53、63、66、464、522题对应文档通过页图分辨率、OCR、crop路径检查 |
| 实际GPU推理与真实题目判分 | 未执行，留给独立小样本实验 |

模拟集成测试包含：多并发、变更副本数续跑、多个已有端点路由、抽取失败只重试抽取、
QA保存后Ctrl+C及恢复、重复SIGINT/SIGHUP清理、GT不进入请求、输入图片变更检测、
快照篡改检测、图片回放、截断/上下文超限/预算耗尽区分、Reasoner类型错误。

真实数据预检发现150条重复OCR记录，内容一致；加载器允许一致重复并拒绝冲突重复。
带小数点的论文文档ID和caption关联另有回归测试。

第一次执行旧14项测试时，中断续跑测试出现一次 Saved result fingerprint mismatch。
没有改动旧代码；该项单独复查通过，随后完整14项复查通过。此处保留记录，不将其归因为新Agent准确率变化。

这些测试验证代码结构和工程行为，不保证新Agent与旧Agent的提示词、检索排序、轨迹或分数等价。
