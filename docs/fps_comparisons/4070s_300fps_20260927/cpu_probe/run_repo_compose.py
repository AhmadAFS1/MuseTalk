"""Run MuseTalk/scripts/benchmark_compose_frame.py unchanged, with torch.load mapped to CPU."""
import sys, runpy, functools, torch
_orig = torch.load
torch.load = functools.partial(_orig, map_location="cpu")
sys.argv = ["benchmark_compose_frame.py"] + sys.argv[1:]
runpy.run_path("/workspace/MuseTalk/scripts/benchmark_compose_frame.py", run_name="__main__")
