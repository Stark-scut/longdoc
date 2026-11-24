#!/usr/bin/env python3
"""Reversible, source-pinned vLLM 0.24.0 backport for Mamba/MM scheduling.

Reference: https://github.com/vllm-project/vllm/pull/51603
The installed version also needs absolute-endpoint alignment and a stop at
the next block boundary after an encoder cap leaves a private partial block.
This is a compatibility backport, not a byte-for-byte application of that PR.
No vLLM import, CUDA initialization, or network access is required.
"""

import argparse
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile

ORIGINAL_SHA256 = "df78076164fe02a63876d7690cefac810f1ba4d8fb58afc15e786c1608064090"
MARKER = "# local-backport: vllm PR 51603 (0.24.0 compatibility)"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one source anchor: {old[:100]!r}")
    return source.replace(old, new, 1)


def transform(source):
    if hashlib.sha256(source.encode()).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("Source differs from the audited vLLM 0.24.0 scheduler; refusing to patch")
    tree = ast.parse(source)
    method = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_mamba_block_aligned_split")
    lines = source.splitlines(keepends=True)
    split = '''    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        num_uncached_common_prefix_tokens: int = 0,
    ) -> int:
        start = (request.num_computed_tokens + num_new_local_computed_tokens
                 + num_external_computed_tokens)
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        if start >= prefill_end:
            return num_new_tokens
        block_size = self.cache_config.block_size
        last_cache_position = request.num_tokens // block_size * block_size
        if self.use_eagle:
            last_cache_position = max(last_cache_position - block_size, 0)
        end = start + num_new_tokens
        if end < prefill_end:
            max_prefill_tokens = self.max_num_scheduled_tokens
            threshold = self.scheduler_config.long_prefill_token_threshold
            if threshold > 0:
                max_prefill_tokens = min(max_prefill_tokens, threshold)
            aligned_end = end // block_size * block_size
            # Keep the normal wait-for-a-fresh-budget policy. Only permit
            # private sub-block progress when a block exceeds a full budget.
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end
        # Encoder caps may leave start inside a block. Resume only up to
        # its next boundary so a reusable state has the correct token index.
        next_boundary = (start // block_size + 1) * block_size
        common_boundary = (
            start + num_uncached_common_prefix_tokens // block_size * block_size
            if num_uncached_common_prefix_tokens >= block_size else 0
        )
        stops = (next_boundary if start % block_size else 0,
                 last_cache_position, common_boundary)
        end = min((point for point in stops if start < point < end), default=end)
        return max(end - start, 0)
'''
    source = "".join(lines[:method.lineno - 1]) + split + "".join(lines[method.end_lineno:])
    running = '''            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

'''
    source = replace_once(source, running, "")
    source = replace_once(source, "\n            # Schedule encoder inputs.\n",
                          "\n            # Apply Mamba alignment before encoder caps.\n" + running
                          + "            # Schedule encoder inputs.\n")
    waiting = '''                # Skip block alignment when setting up async receive (no local work).
                if self.need_mamba_block_aligned_split and not load_kv_async:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                        num_uncached_common_prefix_tokens,
                    )
                    if num_new_tokens == 0:
                        break

'''
    source = replace_once(source, waiting, "")
    aligned_waiting = "\n".join("    " + line if line else "" for line in
                               waiting.splitlines()).rstrip() + "\n\n"
    aligned_waiting = aligned_waiting.replace(
        "# Skip block alignment when setting up async receive (no local work).",
        "# Apply Mamba alignment before encoder caps (local work only).")
    source = replace_once(source, "\n                    # Schedule encoder inputs.\n",
                          "\n" + aligned_waiting + "                    # Schedule encoder inputs.\n")
    source = replace_once(source, '''        lo, hi = get_mm_features_in_window(
            mm_features,
            start=num_computed_tokens,
            end=num_computed_tokens + num_new_tokens + shift_computed_tokens,
        )''', '''        encoder_window_end = num_computed_tokens + num_new_tokens + shift_computed_tokens
        lo, hi = get_mm_features_in_window(
            mm_features,
            start=num_computed_tokens,
            end=encoder_window_end,
        )''')
    source = replace_once(source, '''            end_idx_rel = min(
                num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos
            )''', '''            end_idx_rel = min(num_encoder_tokens, encoder_window_end - start_pos)''')
    source = MARKER + "\n" + source
    compile(source, "patched_scheduler.py", "exec")
    return source


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path, data):
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        shutil.copymode(path, name)
        os.replace(name, path)
        # Avoid stale timestamp-based bytecode after apply/restore.
        for cached in (path.parent / "__pycache__").glob("scheduler.*.pyc"):
            cached.unlink()
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "apply", "restore"))
    parser.add_argument("--scheduler-file", type=Path)
    args = parser.parse_args()
    if args.scheduler_file:
        target = args.scheduler_file.resolve()
    else:
        spec = importlib.util.find_spec("vllm")
        if spec is None:
            parser.error("Use the Python environment containing vLLM")
        target = Path(spec.origin).parent / "v1/core/sched/scheduler.py"
    backup = target.with_suffix(".py.pre_51603")
    current = target.read_bytes()
    if MARKER.encode() in current:
        if not backup.exists() or digest(backup.read_bytes()) != ORIGINAL_SHA256:
            parser.error("Audited original backup missing or changed")
        expected = transform(backup.read_text()).encode()
        if current != expected:
            parser.error("Patched file has additional edits; refusing to overwrite")
        if args.action == "restore":
            atomic_write(target, backup.read_bytes())
            print("[RESTORED]", target)
        else:
            print("[PATCHED]", target)
        return
    if digest(current) != ORIGINAL_SHA256:
        parser.error("Unknown scheduler source; refusing to modify")
    patched = transform(current.decode()).encode()
    if args.action == "apply":
        if backup.exists() and backup.read_bytes() != current:
            parser.error("Existing backup differs; refusing to overwrite")
        if not backup.exists():
            with backup.open("xb") as stream:
                stream.write(current)
        atomic_write(target, patched)
        print("[APPLIED]", target)
        print("Backup:", backup)
        print("Restart vLLM processes using this environment to load the patch.")
    else:
        print("[ORIGINAL]", target, "(compatible; no changes)")


if __name__ == "__main__":
    main()
