# LoRA GRPO 冒烟测试

在已准备好的 qwen35 / ms-swift 环境执行：

```bash
DRY_RUN=1 bash training/grpo/smoke.sh
bash training/grpo/smoke.sh
```

smoke.sh 是原 test_grpo.sh 的归类入口；rewards.py 是其 GSM8K 奖励插件。
默认 GPU 分工、端口和测试预算沿用原脚本，请执行前检查脚本及空闲卡。
默认结果写到仓库 output/，模型可用 MODEL_PATH 覆盖。
此脚本是训练环境检查，与 Native Agent 问答评测不同。
