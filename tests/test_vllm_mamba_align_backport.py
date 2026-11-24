"""CPU regressions executing installed scheduler code without CUDA imports.

The scheduling fragments and encoder cache implementation come from the real
vLLM files. Requests/placeholder positions are small deterministic fixtures.
This checks scheduling and cache progress, not GPU kernels or model accuracy.
"""
import __future__
import ast
from collections import OrderedDict
import copy
import importlib.util
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "backport", ROOT / "evaluations/direct_qa/patch_vllm_mamba_align.py")
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)
spec = importlib.util.find_spec("vllm")
VLLM = Path(spec.origin).parent
TARGET = VLLM / "v1/core/sched/scheduler.py"
ORIGINAL = (TARGET.with_suffix(".py.pre_51603") if patch.MARKER in TARGET.read_text()
            else TARGET).read_text()


def compile_nodes(nodes, scope):
    tree = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(tree, "<installed-vllm-fragment>", "exec",
                 flags=__future__.annotations.compiler_flag), scope)


def window(features, start, end):
    indices = [i for i, f in enumerate(features)
               if f.mm_position.offset < end
               and f.mm_position.offset + f.mm_position.length > start]
    return (min(indices), max(indices) + 1) if indices else (0, 0)


def load_scheduler(source):
    cls = next(n for n in ast.parse(source).body
               if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    cache_cls = next(n for n in ast.parse(
        (VLLM / "v1/core/encoder_cache_manager.py").read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "EncoderCacheManager")
    scope = {"OrderedDict": OrderedDict, "get_mm_features_in_window": window}
    compile_nodes([cache_cls], scope)
    for name in ["_mamba_block_aligned_split", "_try_schedule_encoder_inputs"]:
        compile_nodes([next(n for n in cls.body if getattr(n, "name", None) == name)], scope)
    schedule = next(n for n in cls.body if getattr(n, "name", None) == "schedule")
    running = next(n for n in ast.walk(schedule) if isinstance(n, ast.While)
                   and "req_index" in ast.unparse(n.test))
    indices = [i for i, n in enumerate(running.body)
               if (isinstance(n, ast.Assign) and any(
                   isinstance(t, ast.Name) and t.id == "encoder_inputs_to_schedule"
                   for t in n.targets))
               or (isinstance(n, ast.If) and ast.unparse(n.test) in (
                   "self.need_mamba_block_aligned_split", "request.has_encoder_inputs"))]
    run_nodes = running.body[min(indices):max(indices) + 1]
    # Use the actual waiting-queue branches as well, including async-KV-load skips.
    waiting = next(n for n in ast.walk(schedule) if isinstance(n, ast.While)
                   and any(isinstance(child, ast.If) and ast.unparse(child.test) == "load_kv_async"
                           for child in n.body))
    first = next(i for i, n in enumerate(waiting.body)
                 if isinstance(n, ast.If) and ast.unparse(n.test) == "load_kv_async")
    last = next(i for i, n in enumerate(waiting.body[first:], first)
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "limit_lookahead_tokens"
                        for t in n.targets))
    for name, body in [("running_chunk", run_nodes),
                       ("waiting_chunk", waiting.body[first:last])]:
        wrapper = ast.parse('''def chunk(self, request, num_new_tokens, encoder_compute_budget,
            load_kv_async=False, num_new_local_computed_tokens=0,
            num_external_computed_tokens=0, num_uncached_common_prefix_tokens=0):
    num_computed_tokens = (request.num_computed_tokens + num_new_local_computed_tokens
                           + num_external_computed_tokens)
    token_budget = self.max_num_scheduled_tokens
    defer_prefills = False
    encoder_inputs_to_schedule = None
    external_load_encoder_input = []
    new_encoder_compute_budget = encoder_compute_budget
    while True:
        pass
''').body[0]
        wrapper.name = name
        wrapper.body[-1].body = copy.deepcopy(body) + ast.parse(
            "return encoder_inputs_to_schedule, num_new_tokens, new_encoder_compute_budget").body
        compile_nodes([wrapper], scope)
    methods = {name: scope[name] for name in ("_mamba_block_aligned_split",
               "_try_schedule_encoder_inputs", "running_chunk", "waiting_chunk")}
    return type("SchedulerFixture", (), methods), scope["EncoderCacheManager"]


class Position:
    def __init__(self, offset, length):
        self.offset, self.length = offset, length

    def get_num_embeds(self):
        return self.length

    def get_embeds_indices_in_range(self, start, end):
        return start, end


class Request:
    def __init__(self, positions, tokens=1010):
        self.request_id = "req"
        self.num_computed_tokens = 0
        self.num_prompt_tokens = self.num_tokens = tokens
        self.has_encoder_inputs = bool(positions)
        self.mm_features = [NS(identifier=str(i), mm_position=Position(*p))
                            for i, p in enumerate(positions)]

    def get_num_encoder_embeds(self, i):
        return self.mm_features[i].mm_position.length


def fixture(source, block=768, cache=600, budget=8192, eagle=False):
    Scheduler, Cache = load_scheduler(source)
    s = Scheduler()
    s.cache_config = NS(block_size=block)
    s.scheduler_config = NS(long_prefill_token_threshold=0,
                           disable_chunked_mm_input=False, enable_chunked_prefill=True)
    s.max_num_scheduled_tokens = budget
    s.need_mamba_block_aligned_split = True
    s.use_eagle = eagle
    s.is_encoder_decoder = False
    s.ec_connector = None
    s.encoder_cache_manager = Cache(cache)
    return s


class BackportTests(unittest.TestCase):
    def test_two_images_old_scheduler_stalls_both_paths(self):
        for method in ["running_chunk", "waiting_chunk"]:
            s = fixture(ORIGINAL)
            r = Request([(0, 500), (510, 500)])
            result = getattr(s, method)(r, 1010, 600)
            self.assertTrue(result is None or result[1] == 0)

    def test_patched_scheduler_drains_cache_and_realigns_both_paths(self):
        for method in ["running_chunk", "waiting_chunk"]:
            s = fixture(patch.transform(ORIGINAL))
            r = Request([(0, 500), (510, 500)])
            first = getattr(s, method)(r, 1010, 600)
            self.assertEqual(first[:2], ([0], 510))
            s.encoder_cache_manager.allocate(r, 0)
            r.num_computed_tokens += first[1]
            s.encoder_cache_manager.free_encoder_input(r, 0)
            second = getattr(s, method)(r, 500, 600)
            self.assertEqual(second[:2], ([1], 258))
            s.encoder_cache_manager.allocate(r, 1)
            r.num_computed_tokens += second[1]
            self.assertEqual(r.num_computed_tokens, 768)
            third = getattr(s, method)(r, 242, 600)
            self.assertEqual(third[:2], ([], 242))
            self.assertEqual(r.num_computed_tokens + third[1], r.num_tokens)

    def test_eagle_shift_encodes_image_at_aligned_boundary(self):
        for method in ["running_chunk", "waiting_chunk"]:
            s = fixture(patch.transform(ORIGINAL), block=512, cache=2048, budget=700, eagle=True)
            r = Request([(512, 100)], tokens=1200)
            self.assertEqual(getattr(s, method)(r, 700, 2048)[:2], ([0], 512))

    def test_full_block_budget_policy_and_private_partial_block(self):
        s = fixture(patch.transform(ORIGINAL), block=528, budget=4096)
        r = Request([], tokens=10000)
        self.assertEqual(s._mamba_block_aligned_split(r, 100), 0)
        s.max_num_scheduled_tokens = 256
        self.assertEqual(s._mamba_block_aligned_split(r, 256), 256)
        r.num_computed_tokens = 512
        self.assertEqual(s._mamba_block_aligned_split(r, 256), 16)

    def test_text_decode_and_async_kv_receive(self):
        s = fixture(patch.transform(ORIGINAL))
        r = Request([], tokens=1011)
        r.num_prompt_tokens = r.num_computed_tokens = 1010
        self.assertEqual(s.running_chunk(r, 1, 600)[:2], (None, 1))
        self.assertEqual(s.waiting_chunk(r, 1, 600, load_kv_async=True,
                                        num_external_computed_tokens=768)[:2], (None, 0))

    def test_shared_prefix_and_cached_prefix_offsets(self):
        s = fixture(patch.transform(ORIGINAL), block=528)
        r = Request([], tokens=10000)
        self.assertEqual(s._mamba_block_aligned_split(
            r, 4096, num_uncached_common_prefix_tokens=1200), 1056)
        self.assertEqual(s._mamba_block_aligned_split(
            r, 4096, num_new_local_computed_tokens=600), 456)

    def test_unknown_source_rejected(self):
        with self.assertRaises(ValueError):
            patch.transform(ORIGINAL + "\n# unexpected edit\n")

    def test_varied_image_boundaries_eventually_finish(self):
        rng = random.Random(51603)
        for method in ["running_chunk", "waiting_chunk"]:
            s = fixture(patch.transform(ORIGINAL), block=528, cache=1100)
            for _ in range(40):
                a, b, gap = rng.randint(600, 1000), rng.randint(600, 1000), rng.randint(1, 80)
                s.encoder_cache_manager.reset()
                r = Request([(0, a), (a + gap, b)], tokens=a + gap + b + 5)
                for step in range(30):
                    result = getattr(s, method)(r, r.num_tokens-r.num_computed_tokens, 1100)
                    self.assertIsNotNone(result)
                    inputs, tokens, _ = result
                    self.assertGreater(tokens, 0, (method, a, b, gap, step))
                    for i in inputs or []:
                        s.encoder_cache_manager.allocate(r, i)
                    r.num_computed_tokens += tokens
                    for i in list(s.encoder_cache_manager.get_cached_input_ids(r)):
                        pos = r.mm_features[i].mm_position
                        if pos.offset + pos.length <= r.num_computed_tokens:
                            s.encoder_cache_manager.free_encoder_input(r, i)
                    if r.num_computed_tokens == r.num_tokens:
                        break
                else:
                    self.fail("Request failed to finish within 30 scheduling steps")

    def test_apply_is_idempotent_and_restore_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "scheduler.py"
            target.write_text(ORIGINAL)
            command = [sys.executable, str(Path(patch.__file__))]
            for action in ["check", "apply", "apply", "check"]:
                result = subprocess.run(command + [action, "--scheduler-file", str(target)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(target.read_text(), patch.transform(ORIGINAL))
            result = subprocess.run(command + ["restore", "--scheduler-file", str(target)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(target.read_bytes(), ORIGINAL.encode())


if __name__ == "__main__":
    unittest.main()
