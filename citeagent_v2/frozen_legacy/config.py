#!/usr/bin/env python3
"""Configuration for the primary MMLongBench-Doc experiment."""
import os

# ---------- vLLM 端点(OpenAI 兼容) ----------
# 支持环境变量覆盖:多卡评测时每个分片进程用 CITEAGENT_BASE_URL 指到不同端口。
BASE_URL = os.getenv("CITEAGENT_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("CITEAGENT_API_KEY", "EMPTY")
MODEL = os.getenv("CITEAGENT_MODEL", "Qwen3.5-9B")

# Reasoner 独立端点(可选):设了 CITEAGENT_REASONER_* 时 Reasoner 走独立服务，
# 否则回退到全局模型端点。
REASONER_BASE_URL = os.getenv("CITEAGENT_REASONER_BASE_URL", BASE_URL)
REASONER_API_KEY = os.getenv("CITEAGENT_REASONER_API_KEY", API_KEY)
REASONER_MODEL = os.getenv("CITEAGENT_REASONER_MODEL", MODEL)

# Locator 独立端点(可选):设了 CITEAGENT_LOCATOR_* 时 Locator 工具循环走独立服务，
# 否则回退到全局模型端点。
LOCATOR_BASE_URL = os.getenv("CITEAGENT_LOCATOR_BASE_URL", BASE_URL)
LOCATOR_API_KEY = os.getenv("CITEAGENT_LOCATOR_API_KEY", API_KEY)
LOCATOR_MODEL = os.getenv("CITEAGENT_LOCATOR_MODEL", MODEL)
# Locator 轨迹采集开关:=1 时记录每轮 (messages, teacher assistant) 到轨迹,供 SFT。
# 默认关(正常评测不记,避免 traces 暴涨)。
LOCATOR_TRAJ_DUMP = os.getenv(
    "CITEAGENT_LOCATOR_TRAJ_DUMP", "0"
).strip().lower() in {"1", "true", "yes"}

# ---------- 数据路径 ----------
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.getenv("CITEAGENT_DATA_DIR", os.path.join(_HERE, "..", "data", "mmlongbench")))
JSONL_PATH = os.getenv("CITEAGENT_BLOCKS_PATH", os.path.join(DATA_DIR, "stage2_blocks_cache.jsonl"))
IMAGES_DIR = os.getenv("CITEAGENT_IMAGES_DIR", os.path.join(DATA_DIR, "images"))
SAMPLES_PATH = os.getenv("CITEAGENT_SAMPLES_PATH", os.path.join(DATA_DIR, "samples.json"))
# 同事 crop 好的 image/chart 子图:visual_crops/<doc>/page_<page>.png_b<bid>.jpg
CROP_ROOT = os.getenv("CITEAGENT_CROP_ROOT", os.path.join(DATA_DIR, "visual_crops"))
# 离线为 image/chart crop 生成的 caption；
# 每行 {elem_id, doc_id, page, block_id, caption}。默认仅并入 BM25 检索文本；
# Global Selector 另由 CITEAGENT_GLOBAL_SELECTOR_USE_CROP_CAPTION 控制。
CROP_CAPTIONS_PATH = os.getenv("CITEAGENT_CROP_CAPTIONS_PATH", os.path.join(DATA_DIR, "crop_captions.jsonl"))

# Optional rewrite for OCR files containing machine-specific image paths.
OLD_IMAGE_PREFIX = os.getenv("CITEAGENT_OLD_IMAGE_PREFIX", "")
NEW_IMAGE_PREFIX = os.path.abspath(IMAGES_DIR) + "/"

# ---------- Agent 超参 ----------
MAX_ROUNDS = int(os.getenv("CITEAGENT_MAX_ROUNDS", "15"))  # Locator 最大轮数(可环境变量覆盖)
MAX_MISSING_RETRY = 2          # Reasoner 连续 insufficient 且补检索失败的容忍次数
SEARCH_TOP_K = int(os.getenv("CITEAGENT_SEARCH_TOP_K", "5"))  # search 默认返回页数(可环境变量覆盖)
LOCATOR_TEMPERATURE = float(os.getenv("CITEAGENT_LOCATOR_TEMPERATURE", "0"))      # Locator 工具循环温度(0=贪心，便于复现)
# 推理请求固定 seed:temperature=0 只保证同 logits 取 argmax,但 vLLM 连续批处理下
# batch 组成不同会让 logits 有 1e-6 级浮点抖动,贪心偶尔翻转 token。传 seed 固定
# 采样/tie-break 伪随机,配合 vLLM serve --seed 一起,最大化可复现(仍无法 100% 消除
# 连续批处理的浮点非确定性,故实验须多次取均值±方差)。设为负数则不传 seed。
REQUEST_SEED = int(os.getenv("CITEAGENT_REQUEST_SEED", "42"))
READ_PAGE_MAX = int(os.getenv("CITEAGENT_READ_PAGE_MAX", "3"))              # read_page 单次最多读几页(允许模型自主选页聚合跨页证据)
# Locator 读页上下文实验。latest 保持原行为；persistent_text 会在后续轮次
# 持久展示成功读取过的唯一页面文本。
LOCATOR_READ_CONTEXT = os.getenv(
    "CITEAGENT_LOCATOR_READ_CONTEXT", "persistent_text"
).strip().lower()
# allow:允许 retriever 重读，但持久上下文仍只保留每页一份；skip:程序级跳过已读页。
LOCATOR_REPEAT_READ = os.getenv(
    "CITEAGENT_LOCATOR_REPEAT_READ", "allow"
).strip().lower()
# 首版用字符预算避免为每个评测进程加载 tokenizer。页面按整页原子保留，
# 超预算的旧页退回现有骨架摘要，不截断表格或段落。
LOCATOR_READ_CONTEXT_MAX_CHARS = int(os.getenv(
    "CITEAGENT_LOCATOR_READ_CONTEXT_MAX_CHARS", "24000"
))
# I1-batch:在 persistent_text 下保留最近一次 read_page 的整批页面图。
PERSIST_PAGE_IMAGE = os.getenv(
    "CITEAGENT_PERSIST_PAGE_IMAGE", "0"
).strip().lower() in {"1", "true", "yes"}
GLOBAL_EVIDENCE_SELECTION = os.getenv(
    "CITEAGENT_GLOBAL_EVIDENCE_SELECTION", "0"
).strip().lower() in {"1", "true", "yes"}
# 页级证据:Global Selector 选【证据页】而非元素,整页(截图+全OCR元素+crop)进 E;
# Reasoner 消费页级 E(图文交叉:整页截图→OCR元素,image/chart 的 crop 插在对应 block 位置)。
# 页级链路需 GLOBAL_EVIDENCE_SELECTION=1 一起开(selector 选页是页级 E 的唯一填充来源)。
PAGE_LEVEL_EVIDENCE = os.getenv(
    "CITEAGENT_PAGE_LEVEL_EVIDENCE", "1"
).strip().lower() in {"1", "true", "yes"}
# Global Selector 独立端点(可选):设了 CITEAGENT_GLOBAL_SELECTOR_* 时 selector 走独立
# 服务，否则回退到全局模型端点。
GLOBAL_SELECTOR_BASE_URL = os.getenv("CITEAGENT_GLOBAL_SELECTOR_BASE_URL", BASE_URL)
GLOBAL_SELECTOR_API_KEY = os.getenv("CITEAGENT_GLOBAL_SELECTOR_API_KEY", API_KEY)
GLOBAL_SELECTOR_MODEL = os.getenv("CITEAGENT_GLOBAL_SELECTOR_MODEL", MODEL)
GLOBAL_SELECTOR_MAX_TOKENS = int(os.getenv(
    "CITEAGENT_GLOBAL_SELECTOR_MAX_TOKENS", "8192"
))
GLOBAL_SELECTOR_TIMEOUT = int(os.getenv(
    "CITEAGENT_GLOBAL_SELECTOR_TIMEOUT", "60"
))
GLOBAL_SELECTOR_MAX_RETRY = int(os.getenv(
    "CITEAGENT_GLOBAL_SELECTOR_MAX_RETRY", "1"
))
GLOBAL_SELECTOR_ERROR_BUDGET = int(os.getenv(
    "CITEAGENT_GLOBAL_SELECTOR_ERROR_BUDGET", "2"
))
GLOBAL_SELECTOR_USE_CROP_CAPTION = os.getenv(
    "CITEAGENT_GLOBAL_SELECTOR_USE_CROP_CAPTION", "0"
).strip().lower() in {"1", "true", "yes"}
GLOBAL_SELECTOR_USE_ALL_CROPS = os.getenv(
    "CITEAGENT_GLOBAL_SELECTOR_USE_ALL_CROPS", "0"
).strip().lower() in {"1", "true", "yes"}
# Reasoner 每次都是空上下文重建；输出上限和超时可由评测脚本覆盖。
REASONER_MAX_TOKENS = int(os.getenv("CITEAGENT_REASONER_MAX_TOKENS", "32768"))
# Reasoner 单次请求超时(秒):防个别不收敛难题(如"数图里 icon 个数")无限生成,
# 卡死整条评测分片。超时抛异常,由 reason() 兜底降级 insufficient。
REASONER_TIMEOUT = int(os.getenv("CITEAGENT_REASONER_TIMEOUT", "120"))
# Reasoner chat_json 的总尝试次数(解析失败或超时均计入)。
REASONER_MAX_RETRY = int(os.getenv("CITEAGENT_REASONER_MAX_RETRY", "2"))
REASONER_ERROR_BUDGET = int(os.getenv(
    "CITEAGENT_REASONER_ERROR_BUDGET", "2"
))
# Reasoner 思考模式开关:None=沿用模型默认(不显式传);1/0 = 显式开/关 thinking。
# 页级证据实验里 thinking 常导致图表题生成停不下来顶到超时,可用 =0 关掉做对照。
_rt = os.getenv("CITEAGENT_REASONER_ENABLE_THINKING", "").strip().lower()
REASONER_ENABLE_THINKING = (
    None if _rt == "" else (_rt in {"1", "true", "yes"})
)

# ---------- 多模态开关 ----------
# read_page 是否给模型附上该页整页原图(images/<doc>/page_N.png,原分辨率)。
# True:read_page 回喂文字 OCR 结果后,再补一条带整页原图的 user 消息,
#       让 Locator 对页面有直观理解;False 退回纯 text-first。
READ_PAGE_IMAGE = True

# 是否把 crop caption 并入 BM25 检索文本(修图表检索盲区 Case 5)。
# True:建索引时把 image/chart 元素的 caption 拼进该页检索文本,让"答案只在图里"
#       的图表页也能被内容词/颜色/数值召回。只影响检索,不改元素展示 content。
USE_CROP_CAPTION_IN_SEARCH = True

# ---------- 页号索引约定 ----------
# 我们内部页号 0-indexed(page_0.png 起);samples.json 的 evidence_pages 是 1-indexed。
# 对照 gold 证据页时:our_page = gold_page - PAGE_INDEX_OFFSET
PAGE_INDEX_OFFSET = 1
