# 复用 0 号 GPU 的缓存小测试

当前 GPU 0 对应 http://127.0.0.1:8071/v1，模型名 citeagent-qwen35。测试仅调用现有服务，不部署、不终止服务、不清空缓存。默认 8 次顺序请求，关闭 thinking，每次最多生成 8 tokens。依赖当前 qwen35 环境的 httpx。

~~~bash
python /media/data1/leizh/longdoc/test_vllm_cache.py
~~~

报告写入 output/cache_probe/时间_随机标识.json。--base-url 可换端口；--image 可换本地图片（默认 a_tool_image.png）；--text-only 只测文本；--repeats 3 和 --lines 192 控制重复次数与共同前缀长度。

测试四组情况：
1. text_same：完全相同的长文本请求重复三次。
2. text_changed_suffix：保留长前缀，只改变末尾指令，探测部分前缀复用。
3. image_same：同一张图片和相同长文本请求重复三次。
4. same_image_changed_prefix：图片相同，但图片前方文本靠前位置改变；MM cache 仍可命中，长段 Prefix cache 无法复用。

输出 TTFT（第一个非空 content/reasoning 生成片段的等待时间，包含排队）、总耗时、prompt tokens 和逐请求 cached_tokens。最后打印服务全局 Prefix/MM 查询与命中增量。若有其他评测流量，计数包含它们，耗时也受排队影响；应以逐请求 cached_tokens 判断本请求是否命中。None 表示服务没有返回该字段，不是零。若 metrics 被关闭或不可访问，仍会打印耗时与响应 usage。首次图片可能已经在 MM cache 中，不保证冷缓存；脚本不重置共享服务缓存。

最初的启动命令禁用了 Prefix caching，客户端不能临时开启。当前启动器已改为默认开启 Prefix cache 与 1 GiB MM cache；旧服务需要重启才会应用新默认值。关闭基线可用 --no-enable-prefix-caching。要测试开启效果，请在原评测启动终端 Ctrl+C，让启动器保存进度并退出，再执行：

~~~bash
python /media/data1/leizh/longdoc/citeagent_v2/full_eval/run_native_4gpu.py \
  --gpu-ids 0,1 \
  --output-dir /media/data1/leizh/longdoc/citeagent_v2/runs/native_multi_gpu \
  --enable-prefix-caching
~~~

待 GPU 0 服务就绪后，再开终端运行测试脚本。评测会自动续跑。启动器现已默认开启，显式参数继续兼容；这些参数只影响新启动的服务，开启时还添加 --enable-prompt-tokens-details。若评测全部完成，启动器会直接汇总结果而不部署服务，此时需要单独部署测试服务。

MM cache 指日志中同名的多模态预处理输出缓存（HF processor），按图像项目计数，主要节约 CPU 预处理；它不是语言模型 KV cache，也不能由该命中率推断跳过了所有视觉编码。Prefix cache 复用相同 token/多模态前缀的已计算模型状态（Qwen3.5 为混合注意力模型，还涉及线性注意力状态），主要节约输入 prefill，按 tokens 计数，不缓存最终答案、不直接加速后续逐 token decode。两者独立，可同时命中，是否提高总体速度取决于重复量和实际瓶颈。

当前 vLLM 0.25.1 对混合模型使用模型相关的 Mamba/线性状态缓存模式；块边界和保留状态会影响可复用长度。配置开启不保证命中率 100%；先看完全重复请求，再看改末尾问题的请求，记录真实 cached_tokens 与 mamba_cache_mode。请按本机版本测试，不把其他版本的实验开关直接加入当前环境。

参考：
- https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
- https://docs.vllm.ai/en/latest/design/mm_processing/
