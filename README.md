# LongDoc

Qwen3.5 长文档 Agent、MMLongBench 评测及配套实验。

**新的完整 Agent 在 [agent/](agent/README.md)，统一入口是 `python -m agent`。**
[citeagent_v2/](citeagent_v2/README.md) 是保留的旧实现；新 Agent 不导入它，也不依赖其中的冻结代码或固定基线。

## 从这里开始

```bash
git clone --branch feat/locator-direct-evidence https://github.com/Stark-scut/longdoc.git
cd longdoc
python -m pip install -r agent/requirements.txt
python -m agent --help

# 检查配置、输入路径和部署命令，不启动 GPU、不调用模型、不写结果
python -m agent eval --config configs/agent/native.json \
  --gpu-ids 0,1 --indices 53,63 --dry-run

# 小样本：每张卡部署一份完整模型
python -m agent eval --config configs/agent/native.json \
  --gpu-ids 0,1 --indices 53,63,66,464,522 \
  --concurrency-per-gpu 1 --output-dir output/agent/direct_pages_smoke

# 全量：另用一个输出目录
python -m agent eval --config configs/agent/native.json \
  --gpu-ids 0,1 --concurrency-per-gpu 1 --output-dir output/agent/direct_pages_full
```

根据实际空闲卡修改 GPU 编号。客户端 requirements 不安装 vLLM、PyTorch、Transformers 或训练环境；模型部署需要另外准备兼容的推理环境。已有环境记录在本机 `data/baselines/citeagent_v2/environment.json`，不随此分支的代码克隆。

## 目录导航

| 位置 | 用途 |
| --- | --- |
| [agent/](agent/README.md) | 独立 Native Agent：定位、工具图片、证据、回答、评测、部署和续跑 |
| [configs/agent/native.json](configs/agent/native.json) | 新 Agent 实验配置；路径和角色可用 CLI 覆盖 |
| [evaluations/direct_qa/](evaluations/direct_qa/README.md) | 一次输入整篇文档页图的直接问答评测 |
| [evaluations/answer_extraction/](evaluations/answer_extraction/README.md) | 对已有答案使用 DeepSeek 重提取、重新判分 |
| [examples/](examples/README.md) | 最简 read_page Agent、工具图片、缓存探测 |
| [training/grpo/](training/grpo/README.md) | LoRA GRPO 环境冒烟测试和奖励函数 |
| [tests/](tests/README.md) | 离线自动测试，使用模拟服务与合成图片 |
| [docs/](docs/layout-and-agent.md) | 目录迁移、模块分工、与旧实现的差异 |
| citeagent_v2/ | 旧实现及旧入口，保留用于对照 |
| data/baselines/ | 本机固定基线；Git 忽略 |
| output/ | 全部新旧运行输出；Git 忽略 |

根目录不再保留测试、训练或评测脚本。旧兼容入口和文档链接统一在 [compat/](compat/README.md)，正式实现见上表。

当前Agent流程：Locator调用read_page只收到整页图和页号；通过call_reasoner(pages=[...])直接指定证据页。没有Selector。Reasoner仍收到指定页的整页图、OCR和crop。旧模式输出保持原状，新模式请用新的输出目录。

## 数据与模型

完整 MMLongBench 数据**仍在原位置，没有移动或复制**：

```text
aaai_commit_20260803/aaai_commit/data/mmlongbench/
  samples.json
  stage2_blocks_cache.jsonl
  images_1024/
  visual_crops/
  crop_captions.jsonl
```

新 Agent 默认读取这个相对于仓库的目录，模型默认在仓库同级的 `models/Qwen3.5-9B`。
另一台服务器需单独准备模型和完整数据，然后显式指定：

```bash
python -m agent eval --config configs/agent/native.json \
  --data-dir /path/to/mmlongbench --model-path /path/to/Qwen3.5-9B \
  --gpu-ids 1,4,5 --output-dir output/agent/direct_pages_full
```

新 Agent 的评分脚本和提示词自带，**不需要旧归档中的 Agent 或评分代码**。
旧直接问答/旧 Agent 入口仍保留其原有外部文件要求，见各目录说明。

固定小样本已移到 `data/baselines/citeagent_v2`；旧 Agent 输出已移到 `output/legacy/citeagent_v2`。
本机旧路径 `citeagent_v2/baseline`、`citeagent_v2/runs` 通过兼容链接访问；后者不提交 Git。
完整数据目录与约52GB的 ZIP 保持原位，不入 Git。新服务器克隆本分支不包含旧基线数据或结果；需要复跑旧实验时再单独同步。

API 密钥通过环境变量设置，不写到配置：`CITEAGENT_LOCATOR_API_KEY`、`CITEAGENT_REASONER_API_KEY`、`CITEAGENT_EXTRACTOR_API_KEY`；DeepSeek 重提取使用其脚本约定的环境变量。

## 验证

```bash
python tests/test_agent.py
python tests/test_answer_extraction.py

# 原实现的兼容性检查，需要本机旧基线和旧评测资源
python citeagent_v2/full_eval/tests.py
```

离线测试通过不代表新实现已取得真实模型的相同分数。新 Agent 是独立重写，需在独立输出目录进行真实小样本对照；已有旧评测快照和结果不被改写。
