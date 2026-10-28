# 真实模型示例与探测脚本

这些需要部署模型，和 tests/ 下的离线测试不同。

| 示例 | 启动服务 | 执行 |
| --- | --- | --- |
| 最简 read_page Agent | bash examples/read_page_agent/serve.sh | python examples/read_page_agent/run.py |
| 工具返回图片 | bash examples/tool_image/serve.sh | python examples/tool_image/run.py --image /path/to/a_tool_image.png |
| 缓存探测 | 使用已有服务 | python examples/cache/probe.py --help |

这些是独立小示例。完整 Agent 使用 python -m agent，默认输出在 output/agent/。
旧根目录命令保留为兼容入口，示例的各项参数见对应 README 或 --help。
工具图片原图未包含在仓库中，需单独准备左黄右白原图并传 --image。
