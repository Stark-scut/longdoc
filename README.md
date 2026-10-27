# LongDoc / Qwen3.5 experiments

Qwen3.5 长文档问答、原生工具图片 Agent、MMLongBench 评测和 GRPO 环境测试代码。

## 代码入口

| 入口 | 用途 |
| --- | --- |
| `citeagent_v2/full_eval/run_native_4gpu.py` | Native Agent 全量评测；支持任意 GPU 编号、每卡并发、进度及中断续跑 |
| `citeagent_v2/run.py` | 固定小样本冻结、legacy/native 对照、日志回放与判分 |
| `run_qwen35_mmlongbench_2gpu.py` | 两个单卡模型副本，整篇文档图像直接问答评测 |
| `test_qwen35_read_page_agent.py` | 简单 read_page Agent 测试 |
| `test_qwen35_tool_image.py` | 工具返回图片的链路测试 |
| `test_vllm_cache.py` | Prefix/MM cache 测试 |
| `reextract_mmlongbench_deepseek.py` | 对已有模型答案重新提取短答案、判分 |
| `test_grpo.sh` | LoRA GRPO 环境冒烟测试 |

详细配置见 [Agent 说明](citeagent_v2/README.md)、[全量评测说明](citeagent_v2/full_eval/README.md) 和 [直接问答评测说明](README_qwen35_mmlongbench.md)。

## 克隆与环境

```bash
git clone https://github.com/Stark-scut/longdoc.git
cd longdoc
python -m pip install -r citeagent_v2/requirements.txt
```

上面的 requirements 是 Agent 客户端依赖，不包含 GPU 推理/训练完整环境。另一台服务器还需要安装匹配其驱动和 CUDA 的 PyTorch、vLLM、Transformers；GRPO 另需 ms-swift 等训练依赖。已有实验环境记录见 [environment.json](citeagent_v2/baseline/environment.json)，其中 vLLM=0.25.1、torch=2.11.0+cu130、transformers=5.12.1，仅作迁移参考，不会由客户端 requirements 自动安装。

`citeagent_v2/vendor` 保留运行器依赖的 rank-bm25 0.2.2 及许可证；`frozen_legacy` 和 `baseline` 保留旧代码副本、固定小样本与校验清单。不要直接修改冻结文件以绕过哈希校验。

## 数据和模型需要单独准备

Git 不包含以下本机内容：

- `aaai_commit_20260803/` 和对应 ZIP 归档。
- 根目录 `output/`、`citeagent_v2/runs/` 及临时测试产物。
- Python 缓存及本地密钥配置。
- Qwen3.5 模型权重和完整 MMLongBench 数据集；固定小样本 baseline 中已有的小体积资源保留。

完整评测需要从原服务器单独复制或在新服务器准备：

```text
aaai_commit_20260803/aaai_commit/data/mmlongbench/
  samples.json
  stage2_blocks_cache.jsonl
  images_1024/
  visual_crops/
  crop_captions.jsonl             # 如原实验使用了 caption，应一并复制

aaai_commit_20260803/aaai_commit/MMLongBench-Doc/eval/
  eval_score.py
  prompt_for_answer_extraction.md

/path/to/Qwen3.5-9B/              # 独立准备模型完整目录
```

完整运行器目前仍从上述旧归档位置读取官方评分文件，即使已显式指定数据目录，也需要准备该路径。可以复制旧服务器的两个评分文件；也可以用仓库已冻结的对应文件创建：

```bash
mkdir -p aaai_commit_20260803/aaai_commit/MMLongBench-Doc/eval
cp citeagent_v2/baseline/eval_score.py \
  aaai_commit_20260803/aaai_commit/MMLongBench-Doc/eval/eval_score.py
cp citeagent_v2/baseline/extract_prompt.md \
  aaai_commit_20260803/aaai_commit/MMLongBench-Doc/eval/prompt_for_answer_extraction.md
```

这些目录已被 Git 忽略，准备数据不会把大文件加入提交。部分旧脚本保留了原服务器的绝对路径默认值，使用前按各脚本参数或环境变量覆盖；克隆代码不会自动迁移模型、数据、旧评测结果或 Conda 环境。

## 在新服务器启动 Native Agent

在仓库根目录执行，先修改 GPU、数据目录与模型目录。建议先加 `--indices 53,63` 验证少量题，确认后删除该参数并使用新的输出目录跑完整评测。

```bash
LONGDOC_DATA_DIR="$PWD/aaai_commit_20260803/aaai_commit/data/mmlongbench"
LONGDOC_MODEL_DIR="/path/to/Qwen3.5-9B"

python citeagent_v2/full_eval/run_native_4gpu.py \
  --gpu-ids 0,1 \
  --model-path "$LONGDOC_MODEL_DIR" \
  --input "$LONGDOC_DATA_DIR/samples.json" \
  --blocks "$LONGDOC_DATA_DIR/stage2_blocks_cache.jsonl" \
  --images-dir "$LONGDOC_DATA_DIR/images_1024" \
  --crop-root "$LONGDOC_DATA_DIR/visual_crops" \
  --captions "$LONGDOC_DATA_DIR/crop_captions.jsonl" \
  --output-dir "$PWD/citeagent_v2/runs/native_multi_gpu" \
  --concurrency-per-gpu 1
```

每张卡部署一个完整 BF16 / TP=1 模型，CPU 权重 offload=0；Prefix caching 默认启用，MM processor cache 默认1 GiB。按实际空闲情况指定任意卡号，例如 `--gpu-ids 1,4,5`。使用同一输出目录续跑时保留原题集、输入配置和快照。

先检查部署命令和输入可添加 `--dry-run`；此模式不启动模型、不调用 API。

API 密钥只通过脚本约定的环境变量设置，例如 `DEEPSEEK_API_KEY` 或 `CITEAGENT_LOCATOR_API_KEY`，不要写进代码或提交到 Git。

## 离线验证

```bash
python citeagent_v2/full_eval/tests.py
python tests_reextract_mmlongbench_deepseek.py
```

测试通过验证工程流程，不代表新服务器上已经完成真实 GPU 推理或得到相同评测分数。
