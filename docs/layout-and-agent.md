# 初始仓库整理与独立 Agent（历史版本）

本页记录ef2d427的初始整理。当前流程与根目录清理见 [Locator直接选页](locator-direct-evidence.md)。

分支：chore/repository-layout；整理前提交：4fa02af6e7bc3c4beb36ffef0c0146acfcb2c435。

## 位置与兼容性

新 Agent 全部代码在 agent/，通过 python -m agent 运行。实际依赖为标准库、httpx、Pillow、tqdm；服务端另需兼容的 vLLM/PyTorch/Transformers。
不使用旧系统的动态导入、继承、配置或 vendor。源码包可复制到另一个目录配合显式数据/端点配置运行。

旧 citeagent_v2 的20个核心/冻结程序文件保持原字节，哈希在 reorganization_baseline.json。
小样本 baseline 从 citeagent_v2/baseline 移到 data/baselines/citeagent_v2；
旧 runs 移到 output/legacy/citeagent_v2。旧位置通过符号链接保持访问。
完整数据集和 ZIP 不移动、不复制，仍在 aaai_commit_20260803 原位置。

data/ 与 output/ 不提交 Git。主分支原提交仍保留旧基线文件；本分支克隆后要复跑旧系统，需单独同步 data/baselines/。
已有旧运行快照未修改；旧输出中保存的绝对路径通过本机兼容链接继续可达。

## 根脚本迁移表

| 原入口 | 实际实现 |
| --- | --- |
| evaluate_qwen35_mmlongbench.py | evaluations/direct_qa/evaluate_qwen35_mmlongbench.py |
| run_qwen35_mmlongbench_2gpu.py | evaluations/direct_qa/run_qwen35_mmlongbench_2gpu.py |
| serve_qwen35_mmlongbench.sh | evaluations/direct_qa/serve_qwen35_mmlongbench.sh |
| reextract_mmlongbench_deepseek.py | evaluations/answer_extraction/reextract_mmlongbench_deepseek.py |
| test_qwen35_read_page_agent.py | examples/read_page_agent/run.py |
| serve_qwen35_read_page_agent.sh | examples/read_page_agent/serve.sh |
| test_qwen35_tool_image.py | examples/tool_image/run.py |
| serve_qwen35_tool_image.sh | examples/tool_image/serve.sh |
| test_vllm_cache.py | examples/cache/probe.py |
| test_grpo.sh / gsm8k_plugin.py | training/grpo/smoke.sh / rewards.py |
| tests_reextract_mmlongbench_deepseek.py | tests/test_answer_extraction.py |

旧Python/shell兼容入口和文档符号链接现统一在compat/；新工作维护分类目录中的实际实现。
按文件位置自动推导项目路径的程序已同步修正，GRPO奖励插件与训练脚本保持同目录。
工具图片示例增加 --image 以支持新服务器的图片位置。

## 独立重写与实验边界

本轮不是逐token的等价重构，也不是旧评测的新得分。核心能力保持同类接口，但：
- Locator / Selector / Reasoner 的提示词被独立整理，BM25独立实现。\n- OCR缓存中的相同重复页允许去重；同一页出现冲突OCR则明确报错。文档ID仅去除末尾.pdf，保留论文编号中的小数点。
- read_page 保留整页图、OCR、crop；OCR只渲染一次，不在元数据JSON中重复展开。
- Selector 对相同候选/缺项/正式证据状态复用结果，避免自己的选页操作触发重复选择。
- 不可恢复的结构/类型错误明确终止，避免不断调用耗尽预算的工具后记为 max_rounds。
- JSON格式重试会明确发起纠正请求，全部计入日志和成本。
- Reasoner 答案目前仍要求非空字符串，列表题需要把列表写在字符串内；数组不会自动当作成功答案。
- 当前未加入image-only模式、移除Selector、历史压缩或控制器补调。
- 新输出使用独立schema/指纹，不与旧结果混合续跑。

正式比较必须使用相同题集、图像、模型和预算，分别保存运行配置。
先比较协议有效性、完成率与技术失败，再比较ACC/F1、token、请求数和耗时；中间出现答案不算成功。

## 官方评分来源

agent/metrics/mmlongbench.py 从原固定基线的官方 MMLongBench-Doc eval_score.py 原样复制。
SHA-256：cafa268358b08521cfc9a805c67d86732d943677e4408c5751f336123d09631f。
上游：https://github.com/Mayuyu-Mio/MMLongBench-Doc
它是评分工具，不是旧Agent实现。列表字面量在包装器中先用 ast.literal_eval 安全解析再传入。
官方空列表等边界情况保留为明确的评分错误，不偷偷更改评分公式。GT、gold页号与答案格式不进入在线Agent请求。
