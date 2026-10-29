# 离线自动测试

```bash
python tests/test_agent.py
python tests/test_answer_extraction.py
```

新 Agent 测试使用合成小图片、httpx MockTransport 和本地模拟 HTTP 服务：
原生消息与调用ID、纯图片读页、Locator直接指定证据、无Selector、历史图片、thinking、预算与错误、JSON重试成本、
图片回放、无旧代码依赖、独立快照、配置路径、任意GPU分配、多端口服务、
源码包脱离旧仓库运行、图片变更、并发/续跑、抽取重试、Ctrl+C及重复信号清理、GT隔离。

不部署vLLM，不加载GPU模型，不调用收费API。临时数据和结果放系统临时目录，退出后清理。
旧兼容性测试仍可运行 citeagent_v2/full_eval/tests.py，需要单独准备旧基线及旧评测资源。
