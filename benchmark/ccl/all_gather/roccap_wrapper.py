#!/usr/bin/env python3
import os
import sys

# Get rank from torchrun environment
rank = os.environ.get('RANK', '0')

# Use descriptive name for output files
script_name = "all_gather"

# Hardcoded dispatch filter: only capture persistent_gemm_all_scatter dispatches
DISP_FILTER = "persistent_all_gather/0-"

# Build roccap command: capture --loglevel trace --file <output> --disp <filter> python3 <script> <args>
args = [
    "capture",
    "--loglevel", "trace",
    "--file", f"{script_name}_rank_{rank}.cap",
    "--disp", DISP_FILTER,
    "python3"
] + sys.argv[1:]

# Replace current process with roccap
os.execvp("roccap", ["roccap"] + args)
