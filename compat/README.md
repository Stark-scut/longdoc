# 旧文件名兼容入口

根目录原来的兼容脚本和文档链接统一放在这里，实际实现仍在分类目录。
例如：

```bash
python compat/test_vllm_cache.py --help
python compat/run_qwen35_mmlongbench_2gpu.py --help
bash compat/test_grpo.sh
```

新工作优先使用以下正式入口：

| 任务 | 正式位置 |
| --- | --- |
| 完整Agent | python -m agent |
| 整篇文档直接问答 | evaluations/direct_qa/ |
| 答案重提取 | evaluations/answer_extraction/ |
| 工具图片、简单Agent、缓存示例 | examples/ |
| GRPO | training/grpo/ |
| 离线测试 | tests/ |
| 文档与参考PDF | docs/ |

根目录旧绝对命令路径现已撤下。需要旧名字时，在路径中加compat/；推荐直接使用正式入口。
这些入口根据自身位置定位仓库，可从其他工作目录运行。
