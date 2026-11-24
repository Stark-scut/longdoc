"""CPU-only checks for physical GPU groups and service-level evaluation routing."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from test_agent import Base
from agent.cli import main
from agent.launcher import gpu_ports
from agent.logging import atomic, read


class TensorParallel(Base):
    def argv(self, output):
        path = self.root / "config.json"
        atomic(path, self.config)
        return ["eval", "--config", str(path), "--gpu-ids", "0,4,1,5,2,6,3,7",
                "--tensor-parallel-size", "2", "--concurrency-per-replica", "4",
                "--max-num-seqs", "4", "--output-dir", str(output)]

    def test_group_and_port_validation_including_external_services(self):
        self.assertEqual(gpu_ports("0,4,1,5", None, True, 2), ([0, 4, 1, 5], [8071, 8072]))
        self.assertEqual(gpu_ports("0,1", None, True), ([0, 1], [8071, 8072]))
        self.assertEqual(gpu_ports(None, "9000,9001", False), ([], [9000, 9001]))
        for ids, ports, local, tp in [
                ("0,1,2", None, True, 2), ("0,1", None, True, 0),
                ("0,1", None, True, -1), ("0,0", None, True, 2),
                ("0,1,2,3", "9000,9001,9002,9003", True, 2),
                ("0,1,2,3", "9000,9000", True, 2),
                (None, None, True, 2), (None, "9000", False, 2)]:
            with self.subTest(ids=ids, ports=ports, local=local, tp=tp):
                with self.assertRaises(ValueError):
                    gpu_ports(ids, ports, local, tp)

    def test_dry_run_creates_four_services_instead_of_eight(self):
        output = self.root / "dry"
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(main(self.argv(output) + ["--dry-run"]), 0)
        plan = json.loads(stream.getvalue())
        self.assertEqual(plan["gpu_groups"], [[0, 4], [1, 5], [2, 6], [3, 7]])
        self.assertEqual(plan["ports"], [8071, 8072, 8073, 8074])
        self.assertEqual(plan["tensor_parallel_size"], 2)
        self.assertEqual(plan["concurrency_per_replica"], 4)
        self.assertEqual(len(plan["replicas"]), 4)
        self.assertEqual(sum(r["questions"] for r in plan["replicas"]), 4)
        self.assertEqual(sum(r["documents"] for r in plan["replicas"]), 1)
        for command in plan["commands"]:
            for flag, value in [("--tensor-parallel-size", "2"), ("--max-num-seqs", "4"),
                                ("--dtype", "bfloat16"), ("--cpu-offload-gb", "0")]:
                self.assertEqual(command[command.index(flag) + 1], value)
        self.assertFalse(output.exists())

    def test_local_launch_masks_gpu_groups_and_routes_workers_to_four_replicas(self):
        output = self.root / "launch"
        calls = []
        def popen(command, **kwargs):
            server = "vllm.entrypoints.openai.api_server" in command
            process = SimpleNamespace(pid=17000 + len(calls), returncode=None if server else 0,
                                      poll=lambda: None if server else 0)
            calls.append((command, kwargs, process))
            return process
        with patch("agent.launcher.subprocess.Popen", side_effect=popen), \
                patch("agent.launcher.port_available", return_value=True), \
                patch("agent.launcher.check_gpus") as check, \
                patch("agent.launcher.wait_ready") as ready, \
                patch("agent.launcher.stop_groups") as stop, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(self.argv(output)), 0)
        check.assert_called_once_with([0, 4, 1, 5, 2, 6, 3, 7])
        servers = [c for c in calls if "vllm.entrypoints.openai.api_server" in c[0]]
        self.assertEqual(len(servers), 4)
        self.assertEqual([kwargs["env"]["CUDA_VISIBLE_DEVICES"] for _, kwargs, _ in servers],
                         ["0,4", "1,5", "2,6", "3,7"])
        self.assertTrue(all(kwargs["start_new_session"] for _, kwargs, _ in calls))
        self.assertEqual(ready.call_args.args[1], [8071, 8072, 8073, 8074])
        runtime = read(output / "runtime.json")
        self.assertEqual(runtime["gpu_groups"], [[0, 4], [1, 5], [2, 6], [3, 7]])
        self.assertEqual(runtime["tensor_parallel_size"], 2)
        self.assertTrue(runtime["map_replicas"])
        self.assertEqual(set(runtime["assignments"]), {f"{i}:{j}" for i in range(4) for j in range(4)})
        self.assertEqual(len(runtime["queues"]), 4)
        for i, group in enumerate(runtime["gpu_groups"]):
            record = read(output / f"server{i}/launch.json")
            self.assertEqual(record["gpu_ids"], group)
            self.assertEqual(record["tensor_parallel_size"], 2)
        self.assertEqual(len(stop.call_args_list[1].args[0]), 4)

    def test_3090_script_forwards_paths_and_smoke_overrides(self):
        path = self.root / "config.json"; atomic(path, self.config)
        output = self.root / "script_dry"
        env = os.environ.copy()
        env.update(PYTHON=sys.executable, MODEL_PATH=self.config["model_path"], DATA_DIR=str(self.data),
                   OUTPUT_DIR=str(output), GPU_IDS="0,1,2,3,4,5,6,7", CONCURRENCY="4")
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(["bash", str(root / "evaluations/agent/run_3090.sh"), "--config", str(path),
                                 "--limit", "2", "--dry-run"], cwd=self.root, env=env,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["total"], 2)
        self.assertEqual(plan["gpu_groups"], [[0, 1], [2, 3], [4, 5], [6, 7]])
        self.assertEqual(plan["config"]["max_model_len"], 65536)
        self.assertEqual(plan["config"]["server"]["mm_processor_cache_gb"], 3)
        self.assertEqual(plan["config"]["server"]["max_num_batched_tokens"], 4096)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    import unittest
    unittest.main()
