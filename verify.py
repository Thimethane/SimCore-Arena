"""
verify.py — End-to-end sanity check for SimCore Arena
=============================================================

Runs every critical subsystem without a display server and reports
PASS / FAIL for each check. Run this after a fresh clone to confirm
the environment is correctly set up.

Usage
-----
    python verify.py
"""

import sys
import math
import random
import importlib
import traceback
import time

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
INFO = "\033[94m  INFO\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    try:
        note = fn()
        results.append((name, True, note or ""))
        print(f"{PASS}  {name}" + (f"  ({note})" if note else ""))
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"{FAIL}  {name}")
        print(f"        {e}")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("  SimCore Arena — Verification")
print("═" * 60 + "\n")

# ── 1. Python version ─────────────────────────────────────────────────────────
def chk_python():
    v = sys.version_info
    assert v >= (3, 10), f"Python ≥ 3.10 required, got {v.major}.{v.minor}"
    return f"{v.major}.{v.minor}.{v.micro}"
check("Python ≥ 3.10", chk_python)

# ── 2. Dependencies ───────────────────────────────────────────────────────────
for pkg in ("panda3d", "matplotlib", "numpy", "PIL", "pybind11"):
    def chk_pkg(p=pkg):
        mod = importlib.import_module(p)
        ver = getattr(mod, "__version__", "?")
        return ver
    check(f"Import {pkg}", chk_pkg)

# ── 3. C++ module ─────────────────────────────────────────────────────────────
def chk_cpp():
    import proximity_cpp
    pairs = proximity_cpp.find_neighbors([0.0, 1.0, 10.0], [0.0, 1.0, 10.0], 3.0)
    assert (0, 1) in pairs, "Expected pair (0,1) within radius 3"
    assert (0, 2) not in pairs, "Pair (0,2) should be outside radius"
    return f"{len(pairs)} pair(s) found"
check("C++ proximity_cpp module", chk_cpp)

# ── 4. C++ correctness (Python vs C++ must agree) ────────────────────────────
def chk_correctness():
    import proximity_cpp
    random.seed(99)
    n, r = 50, 6.0
    xs = [random.uniform(-24, 24) for _ in range(n)]
    ys = [random.uniform(-24, 24) for _ in range(n)]

    # Reference: Python
    py_pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = xs[i]-xs[j], ys[i]-ys[j]
            if dx*dx + dy*dy < r*r:
                py_pairs.add((i, j))

    cpp_pairs = set(proximity_cpp.find_neighbors(xs, ys, r))
    diff = py_pairs.symmetric_difference(cpp_pairs)
    assert not diff, f"Mismatch: {diff}"
    return f"{len(py_pairs)} pairs verified"
check("Python == C++ output (n=50, r=6.0)", chk_correctness)

# ── 5. Agent FSM ──────────────────────────────────────────────────────────────
def chk_agent_fsm():
    # Inline minimal agent
    class V:
        def __init__(self, x, y): self.x, self.y = x, y

    AVOID_R, SEEK_R = 4.0, 12.0

    class A:
        def __init__(self, x, y, tx, ty):
            self.pos    = V(x, y)
            self.vel    = V(1.0, 0.0)
            self.target = V(tx, ty)
            self.state  = "IDLE"
            self._wangle = 0.0

        def decide(self, nb):
            for n in nb:
                d = math.hypot(self.pos.x - n.pos.x, self.pos.y - n.pos.y)
                if d < AVOID_R:
                    self.state = "AVOID"; return
            d_tgt = math.hypot(self.pos.x - self.target.x, self.pos.y - self.target.y)
            if d_tgt < SEEK_R:
                self.state = "SEEK"; return
            self.state = "IDLE"

    # AVOID: close neighbour
    a1 = A(0, 0, 100, 100)
    a2 = A(2, 0, 100, 100)   # within AVOID_R=4
    a1.decide([a2])
    assert a1.state == "AVOID", f"Expected AVOID, got {a1.state}"

    # SEEK: no neighbours, target close
    a3 = A(0, 0, 5, 0)       # target at dist 5 < SEEK_R=12
    a3.decide([])
    assert a3.state == "SEEK", f"Expected SEEK, got {a3.state}"

    # IDLE: no neighbours, target far
    a4 = A(0, 0, 50, 0)
    a4.decide([])
    assert a4.state == "IDLE", f"Expected IDLE, got {a4.state}"

    # Priority: AVOID wins even when target is close
    a5 = A(0, 0, 5, 0)
    a6 = A(2, 0, 5, 0)
    a5.decide([a6])
    assert a5.state == "AVOID", f"AVOID priority failed, got {a5.state}"

    return "AVOID > SEEK > IDLE priority confirmed"
check("Agent FSM state priority", chk_agent_fsm)

# ── 6. Proximity system Python ────────────────────────────────────────────────
def chk_prox_python():
    sys.path.insert(0, ".")
    from systems.proximity import ProximitySystemPython, PROXIMITY_RADIUS
    from panda3d.core import Vec3, Point3
    from agents.agent import Agent

    random.seed(0)
    agents = [Agent(i, Vec3(random.uniform(-5,5), random.uniform(-5,5), 0),
                    Point3(0,0,0)) for i in range(10)]
    ps = ProximitySystemPython()
    nb = ps.get_neighbors(agents, radius=6.0)
    assert set(nb.keys()) == {a.agent_id for a in agents}
    return f"10 agents, {sum(len(v) for v in nb.values())//2} mutual pairs"
check("ProximitySystemPython", chk_prox_python)

# ── 7. Proximity system C++ ───────────────────────────────────────────────────
def chk_prox_cpp():
    from systems.proximity import ProximitySystemCpp
    from panda3d.core import Vec3, Point3
    from agents.agent import Agent

    random.seed(0)
    agents = [Agent(i, Vec3(random.uniform(-5,5), random.uniform(-5,5), 0),
                    Point3(0,0,0)) for i in range(10)]
    ps = ProximitySystemCpp()
    assert ps.cpp_available, "C++ module not loaded"
    nb = ps.get_neighbors(agents, radius=6.0)
    assert set(nb.keys()) == {a.agent_id for a in agents}
    return f"10 agents, {sum(len(v) for v in nb.values())//2} mutual pairs"
check("ProximitySystemCpp", chk_prox_cpp)

# ── 8. SimulationSystem ───────────────────────────────────────────────────────
def chk_sim():
    from systems.simulation import SimulationSystem
    sim = SimulationSystem(num_agents=20, mode="python")
    assert sim.agent_count == 20
    before = [(a.position.x, a.position.y) for a in sim.agents]
    for _ in range(10):
        sim.update(1/60)
    after = [(a.position.x, a.position.y) for a in sim.agents]
    moved = sum(1 for b, a in zip(before, after) if b != a)
    assert moved > 0, "No agents moved after 10 steps"
    return f"{moved}/20 agents moved"
check("SimulationSystem (python, 20 agents, 10 steps)", chk_sim)

# ── 9. Simulation determinism ─────────────────────────────────────────────────
def chk_determinism():
    from systems.simulation import SimulationSystem

    def run(seed):
        random.seed(seed)
        sim = SimulationSystem(num_agents=15, mode="python")
        for _ in range(30):
            sim.update(1/60)
        return [(round(a.position.x, 6), round(a.position.y, 6))
                for a in sim.agents]

    r1 = run(7)
    r2 = run(7)
    assert r1 == r2, "Simulation is not deterministic!"
    return "same seed → same result"
check("Simulation determinism (same seed)", chk_determinism)

# ── 10. Benchmark module import ───────────────────────────────────────────────
def chk_benchmark():
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("benchmark",
               os.path.join(os.path.dirname(__file__), "benchmark.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    results_b = mod.run_benchmark([5], num_frames=30, use_cpp=False)
    assert 5 in results_b
    assert results_b[5]["fps"] > 0
    return f"5 agents, {results_b[5]['avg_ms']:.3f} ms/frame"
check("benchmark.py (5 agents, 30 frames)", chk_benchmark)

# ── 11. C++ speedup at scale ──────────────────────────────────────────────────
def chk_speedup():
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("benchmark",
               os.path.join(os.path.dirname(__file__), "benchmark.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import proximity_cpp

    py_r  = mod.run_benchmark([40], num_frames=100, use_cpp=False)
    cpp_r = mod.run_benchmark([40], num_frames=100, use_cpp=True, cpp_mod=proximity_cpp)
    speedup = py_r[40]["avg_ms"] / cpp_r[40]["avg_ms"]
    assert speedup > 1.0, f"Expected C++ speedup > 1×, got {speedup:.2f}×"
    return f"C++ is {speedup:.2f}× faster at 40 agents"
check("C++ speedup > 1× (40 agents, 100 frames)", chk_speedup)

# ── Summary ───────────────────────────────────────────────────────────────────
total  = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed

print(f"\n{'═'*60}")
print(f"  Results: {passed}/{total} passed", end="")
if failed:
    print(f"  —  \033[91m{failed} FAILED\033[0m")
else:
    print(f"  —  \033[92mAll checks passed ✓\033[0m")
print("═" * 60 + "\n")

sys.exit(0 if failed == 0 else 1)
