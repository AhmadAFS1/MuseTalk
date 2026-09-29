"""20 Hz CPU-busy sampler for a live run: <RUN>/cpu_hi.csv = t_mono, busy fraction of the client CPU set, of the
server CPU set, and of the busiest single client CPU, per 50 ms. Stops when <RUN>/STOP_SAMPLER exists.
  cpu_sampler.py <RUN> [client cpus, default 12-15,28-31] [server cpus, default 0-11,16-27]"""
import os
import sys
import time


def cpus(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def read():
    t = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                p = line.split()
                v = list(map(int, p[1:9]))
                t[int(p[0][3:])] = (sum(v), v[3] + v[4])  # total, idle+iowait
    return t


run = sys.argv[1]
cl = cpus(sys.argv[2] if len(sys.argv) > 2 else "12-15,28-31")
sv = cpus(sys.argv[3] if len(sys.argv) > 3 else "0-11,16-27")
prev = read()
with open(os.path.join(run, "cpu_hi.csv"), "w") as out:
    out.write("t_mono,client_busy,server_busy,client_max_cpu_busy\n")
    while not os.path.exists(os.path.join(run, "STOP_SAMPLER")):
        time.sleep(0.05)
        cur = read()
        busy = {}
        for c in cur:
            dt = cur[c][0] - prev[c][0]
            busy[c] = 0.0 if dt <= 0 else 1.0 - (cur[c][1] - prev[c][1]) / dt
        prev = cur
        cb = sum(busy[c] for c in cl if c in busy) / len(cl)
        sb = sum(busy[c] for c in sv if c in busy) / len(sv)
        out.write(f"{time.monotonic():.3f},{cb:.3f},{sb:.3f},{max(busy[c] for c in cl if c in busy):.3f}\n")
        out.flush()
