# CiteAgent 独立上下文修复实验

此目录实现审计方案的前四步。原仓库的 Python 文件不改动；frozen_legacy 是逐字节复制的旧实现，只读保留。新逻辑在 agent.py、transport.py、runtime.py 与 run.py 中。

## 已实现的内容

1. 冻结源代码哈希、有效配置、模型 / 模板元数据、固定题集、对应页面/OCR/crop、归档旧结果及原始轨迹。启动时验证冻结文件，拒绝混用不同配置的输出。
2. native 模式按真实 assistant → tool 续接历史，保留 reasoning、tool_calls 和调用 ID；逐调用返回真实图片，不重建成普通 user；完整记录结束原因和 usage。
3. 区分执行 completed / incomplete / error、Reasoner answerable / insufficient、技术错误、输出截断和预算耗尽。最终答案只取有效 Reasoner 返回。框架兜底默认关闭，显式开启后单独记录。旧模式保留其原来的兜底策略。
4. 提供离线结构/错误路径测试、固定小样本 legacy/native 对照、官方短答案抽取与判分、请求回放、断点续跑及对比报告。工具链成功和 QA 得分分别统计。

未实现审计方案第五步：不做历史压缩、不删除历史图片、不迁移训练数据。长上下文超限会明确失败，不偷偷截页。

## 目录

~~~text
frozen_legacy/          原 citeagent/*.py 的不变副本
source_manifest.json   原文件与副本的 SHA-256
baseline/              已冻结的 4 题、输入资源、模型元数据、归档结果
agent.py               新的原生工具对话与状态控制
transport.py           显式超时/重试/thinking、请求及响应记录
runtime.py             只加载新目录中的旧代码副本
run.py                 freeze/run/compare/verify/doctor/replay/validate
tests.py               无模型的功能与回归测试
serve.sh               可选：单卡 BF16 / TP=1 的工具调用服务
vendor/                此目录内安装的 rank-bm25，不修改 conda 环境
runs/                  新实验输出，和归档结果分开
~~~

## 默认冻结实验

固定原始题号 281、283、289、291，来自两份 10 页 / 9 页短文档。使用 images_1024、原 OCR 和已有 crop；内部页号沿用旧系统的 0-based，离线证据页比较时再加 1。

默认 Locator thinking=true、Reasoner/Selector/extractor thinking=false；Locator 8 轮、一次最多读 3 页、输出 8192 tokens，Reasoner 8192、Selector 4096、抽取 2048；每次 HTTP 最多等 120 秒。native 不合并或删除历史。

单题 900 秒为请求之间检查的 wall budget，并限制下一次 HTTP 超时，不是强制中断线程的硬时限。服务未返回 usage 的失败请求无法精确计 token，日志和报告明确保留这个限制。

原配置另存 original_effective_config.json。受控复跑的生效参数另存 settings.json，不能将其误称为对历史 H800 检查点的精确重现：历史模型权重和模板身份不完整，归档分数仅作参考。默认基础模型路径为 /media/data1/leizh/models/Qwen3.5-9B；记录配置/模板哈希和权重分片大小，不读取并哈希全部 18 GB 权重。

## 使用方式

在 qwen35 环境中执行：

~~~bash
cd /media/data1/leizh/longdoc/citeagent_v2
python run.py verify
python run.py validate
~~~

已有 baseline 不重复创建。要更换题集、模型或预算，创建新的冻结目录：

~~~bash
python run.py freeze \
  --baseline baseline_other \
  --indices 281,283,289,291 \
  --url http://127.0.0.1:8067/v1 \
  --model citeagent-qwen35 \
  --thinking \
  --no-reasoner-thinking \
  --max-rounds 8 \
  --read-page-max 3
~~~

单独的 --locator-url/--locator-model、--reasoner-url/--reasoner-model、--selector-url/--selector-model、--extractor-url/--extractor-model 可以指定各组件服务。访问密钥只从 CITEAGENT_LOCATOR_API_KEY 等对应环境变量读取，不写入冻结配置和 HTTP 日志。

如果需要部署服务，先选一张确实空闲的卡，再运行；脚本不会自动选择卡或终止已有服务：

~~~bash
GPU_ID=0 PORT=8067 bash serve.sh
~~~

GPU_ID=0 仅为命令示例。该服务是一张卡放完整 BF16 模型，TP=1，CPU offload=0，关闭 prefix caching，保留 eager。必须有工具解析器 qwen3_coder、推理解析器 qwen3 和 --enable-auto-tool-choice。直接使用已有直接问答评测服务前，应确认这些能力，不要只看 /health。

确认冻结配置指向的模型服务可达：

~~~bash
python run.py doctor
~~~

在同一服务上串行跑两组，避免 GPU 调度相互干扰：

~~~bash
python run.py compare --output runs/small_comparison
~~~

只跑一个模式：

~~~bash
python run.py run --mode legacy --output runs/legacy
python run.py run --mode native --output runs/native
~~~

分开生成答案与离线评分：

~~~bash
python run.py compare --stage agent --output runs/small_comparison
python run.py compare --stage score --output runs/small_comparison
~~~

重复相同命令复用已保存结果和已完成的抽取；加 --retry-failed 才重新运行未完成/失败的 Agent 样本。每题生成结果先落盘，随后才抽取，抽取失败不会丢掉推理答案。

比较期间可查看 legacy.log、native.log 和每题 attempt_*.jsonl；每题结束都会打印样本号、状态、答案、分数和耗时。

## 输出和评估口径

~~~text
runs/small_comparison/
  legacy.log
  native.log
  legacy/
    sample_0281/result.json
    sample_0281/attempt_001.jsonl
    sample_0281/assets/<sha256>.png
    summary.json
    results.json
  native/...
  comparison.json
  comparison.md
~~~

result.json 区分：

- execution_status：completed / incomplete / error。
- semantic_status：成功的 Reasoner 语义判断；技术失败不伪装成 insufficient。
- stop_reason：reasoner_answerable、max_rounds、premature_final、timeout、context_limit、output_truncated 等。
- raw_answer：Reasoner 原始答案，和抽取答案 pred 分开。
- scoring_status：scored / not_answered / error。
- controller_fallback_calls 与 finalized_by：是否发生框架补调。
- cost / agent_cost：角色请求次数、耗时、服务报告的 token，以及失败请求数。
- evidence_version：避免在证据更新后接受旧的充分性判断。
- last_tool_observation_consumed_by_locator：到达预算时，最后一批观察是否被后续 Locator 消费。Reasoner 已给有效答案时无需额外生成 Locator 收尾文本，报告会注明。

官方 eval_score.py 和短答案抽取 prompt 也冻结进 baseline。GT 只在离线 score 阶段使用，不输入在线 Agent / Selector / Reasoner / extractor。未完成或技术失败在固定题集分母内按零计，即使 GT 是 Not answerable 也不奖励。这与旧入口把所有 abstain 转成 Not answerable 的做法有区别；legacy_raw_result、legacy_benchmark_response 和归档结果都保留，方便区分。

两组对比包含消息、历史以及错误/终止语义修复，不是“只改 role 字段”的单因素实验。4 道短文档题用于调试和回归，不能据此推断完整 MMLongBench 成绩。

## 回放与日志

日志保留所有实际请求、返回、reasoning、finish_reason、usage、工具返回和控制事件。图片字节存成内容寻址资源，日志中用 asset:// 引用，不重复存 base64。回放检查图片哈希，重建真正的 data URL；不会向服务提交：

~~~bash
python run.py replay \
  --trace runs/small_comparison/native/sample_0281/attempt_001.jsonl \
  --sequence 2 \
  --output runs/restored_request.json
~~~

sequence 必须对应日志中的 llm_request 事件。序号 2 仅为示例。

## 验证与限制

离线测试使用 unittest、httpx.MockTransport 和合成模型返回，验证真实的新主循环、旧会话路径、图片回放、官方判分及错误分支。测试通过不代表真实模型答对；真实 GPU 对照结果只有在服务可用且执行 compare 后才产生。

部署服务、live doctor 和真实小样本对照可能需要当前执行环境的网络/GPU权限；冻结与离线测试不会加载 GPU 模型。baseline 内保存历史结果不代表新代码已经得到相同分数。

依赖 rank-bm25 固定为 0.2.2，安装在 vendor 中；其余依赖沿用 qwen35 已有包。若 vendor 缺失，可仅补装在此目录：

~~~bash
python -m pip install --no-cache-dir --no-deps --no-compile \
  --target vendor rank-bm25==0.2.2
~~~
