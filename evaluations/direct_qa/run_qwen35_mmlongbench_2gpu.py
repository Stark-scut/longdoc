#!/usr/bin/env python3
"""Compatibility name; use run_replicas.py for any GPU count."""
from run_replicas import *
import sys

if __name__ == "__main__":
    argv = sys.argv[1:]
    for option, value in {"--gpu-ids":"1,2", "--max-model-len":"131072",
                          "--max-num-seqs":"1", "--max-num-batched-tokens":"1024",
                          "--mm-processor-cache-gb":"0"}.items():
        if not any(arg == option or arg.startswith(option+"=") for arg in argv):
            argv.extend([option,value])
    if not any(arg in {"--enable-prefix-caching","--no-enable-prefix-caching"} for arg in argv):
        argv.append("--no-enable-prefix-caching")
    if not any(arg in {"--truncate-long-documents","--no-truncate-long-documents"} for arg in argv):
        argv.append("--no-truncate-long-documents")
    raise SystemExit(main(argv))
