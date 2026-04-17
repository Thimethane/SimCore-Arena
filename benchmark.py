"""
benchmark.py — Headless performance benchmark for SimCore Arena
======================================================================

Runs the simulation loop (without Panda3D rendering) for a fixed number
of frames across multiple agent counts, comparing Python vs C++ proximity
back-ends.

Usage
-----
    python benchmark.py                    # default settings
    python benchmark.py --frames 300
    python benchmark.py --agents 10 30 60 --frames 300
    python benchmark.py --output results/bench.png

Output
------
  • Console table with per-configuration results
  • benchmark.png — line chart comparing Python vs C++ frame times
"""

import argparse
import math
import random
import time
import sys
import os

import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Minimal agent/proximity implementation (no Panda3D needed) ────────────────
# We replicate only what the benchmark exercises so that the benchmark runs
# cleanly even without a display server.

AVOID_RADIUS   = 4.0
SEEK_RADIUS    = 12.0
MAX_SPEED      = 5.0
IDLE_SPEED     = 2.0
WANDER_CHANGE  = 0.02
ARENA_HALF     = 24.0
PROXIMITY_RADIUS = 6.0


class _Vec3:
    __slots__ = ("x", "y", "z")
    def __init__(self, x=0., y=0., z=0.):
        self.x, self.y, self.z = x, y, z


class _Agent:
    """Lightweight agent replica — no Panda3D dependency."""

    def __init__(self, agent_id, x, y):
        self.agent_id = agent_id
        self.pos  = _Vec3(x, y, 0.)
        self.vel  = _Vec3(random.uniform(-1,1), random.uniform(-1,1), 0.)
        self._normalise(IDLE_SPEED)
        self._wangle = random.uniform(0, 6.2832)
        self.state = "IDLE"
        self.target = _Vec3(0., 0., 0.)

    def update(self, dt, neighbors):
        self._decide(neighbors)
        if self.state == "SEEK":
            self._seek(dt)
        elif self.state == "AVOID":
            self._avoid(dt, neighbors)
        else:
            self._idle(dt)
        self._clamp()

    def _decide(self, neighbors):
        for nb in neighbors:
            if self._dist(nb.pos) < AVOID_RADIUS:
                self.state = "AVOID"
                return
        tx = self.target.x - self.pos.x
        ty = self.target.y - self.pos.y
        if math.sqrt(tx*tx + ty*ty) < SEEK_RADIUS:
            self.state = "SEEK"
            return
        self.state = "IDLE"

    def _seek(self, dt):
        dx = self.target.x - self.pos.x
        dy = self.target.y - self.pos.y
        ln = math.hypot(dx, dy)
        if ln < 0.01: return
        self.pos.x += dx/ln * MAX_SPEED * dt
        self.pos.y += dy/ln * MAX_SPEED * dt

    def _avoid(self, dt, neighbors):
        rx = ry = 0.
        for nb in neighbors:
            d = self._dist(nb.pos)
            if 0.01 < d < AVOID_RADIUS:
                w = (AVOID_RADIUS - d) / AVOID_RADIUS
                rx += (self.pos.x - nb.pos.x) / d * w
                ry += (self.pos.y - nb.pos.y) / d * w
        ln = math.hypot(rx, ry)
        if ln > 0.01:
            self.pos.x += rx/ln * MAX_SPEED * dt
            self.pos.y += ry/ln * MAX_SPEED * dt

    def _idle(self, dt):
        if random.random() < WANDER_CHANGE:
            self._wangle += random.uniform(-0.8, 0.8)
        self.pos.x += math.cos(self._wangle) * IDLE_SPEED * dt
        self.pos.y += math.sin(self._wangle) * IDLE_SPEED * dt

    def _clamp(self):
        if self.pos.x >  ARENA_HALF: self.pos.x =  ARENA_HALF
        if self.pos.x < -ARENA_HALF: self.pos.x = -ARENA_HALF
        if self.pos.y >  ARENA_HALF: self.pos.y =  ARENA_HALF
        if self.pos.y < -ARENA_HALF: self.pos.y = -ARENA_HALF

    def _dist(self, other):
        dx = self.pos.x - other.x
        dy = self.pos.y - other.y
        return math.sqrt(dx*dx + dy*dy)

    def _normalise(self, spd):
        ln = math.hypot(self.vel.x, self.vel.y)
        if ln > 0.01:
            self.vel.x = self.vel.x / ln * spd
            self.vel.y = self.vel.y / ln * spd


def _proximity_python(agents, radius):
    """Pure-Python O(n²) neighbour scan."""
    neighbors = {a.agent_id: [] for a in agents}
    n = len(agents)
    for i in range(n):
        ai = agents[i]
        for j in range(i+1, n):
            aj = agents[j]
            dx = ai.pos.x - aj.pos.x
            dy = ai.pos.y - aj.pos.y
            if dx*dx + dy*dy < radius*radius:
                neighbors[ai.agent_id].append(aj)
                neighbors[aj.agent_id].append(ai)
    return neighbors


def _proximity_cpp(agents, radius, mod):
    """C++ accelerated neighbour scan."""
    ids = [a.agent_id for a in agents]
    xs  = [a.pos.x for a in agents]
    ys  = [a.pos.y for a in agents]
    pairs = mod.find_neighbors(xs, ys, radius)
    id_map = {a.agent_id: a for a in agents}
    neighbors = {aid: [] for aid in ids}
    for i, j in pairs:
        neighbors[ids[i]].append(id_map[ids[j]])
        neighbors[ids[j]].append(id_map[ids[i]])
    return neighbors


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(
    agent_counts: list[int],
    num_frames: int,
    use_cpp: bool,
    cpp_mod=None,
) -> dict[int, dict]:
    """
    Run the headless simulation for each agent count.

    Returns
    -------
    dict[num_agents -> {"total_s": float, "avg_ms": float, "fps": float}]
    """
    results = {}
    mode_label = "C++" if use_cpp else "Python"

    for n in agent_counts:
        random.seed(42)           # reproducible spawn positions
        agents = [
            _Agent(i,
                   random.uniform(-ARENA_HALF*0.8, ARENA_HALF*0.8),
                   random.uniform(-ARENA_HALF*0.8, ARENA_HALF*0.8))
            for i in range(n)
        ]

        dt = 1.0 / 60.0           # fixed time step for reproducibility
        frame_times = []

        for _ in range(num_frames):
            t0 = time.perf_counter()

            if use_cpp and cpp_mod:
                nb_map = _proximity_cpp(agents, PROXIMITY_RADIUS, cpp_mod)
            else:
                nb_map = _proximity_python(agents, PROXIMITY_RADIUS)

            for agent in agents:
                agent.update(dt, nb_map[agent.agent_id])

            frame_times.append(time.perf_counter() - t0)

        total_s  = sum(frame_times)
        avg_ms   = total_s / num_frames * 1000
        fps_val  = num_frames / total_s

        results[n] = {
            "total_s": total_s,
            "avg_ms":  avg_ms,
            "fps":     fps_val,
        }
        print(f"  [{mode_label:6s}] {n:3d} agents | "
              f"total={total_s:6.3f}s | avg={avg_ms:6.3f} ms | fps={fps_val:7.1f}")

    return results


# ── SimCore Arena palette (benchmark chart) ───────────────────────────────────
_C = {
    "bg_fig":    "#0b0c0e",   # near-black canvas
    "bg_ax":     "#111318",   # plot area
    "grid_maj":  "#1c1f26",
    "grid_min":  "#14161b",
    "border":    "#2e3340",
    "text":      "#e8dcc8",   # warm off-white
    "py_line":   "#e8820c",   # amber-orange  (Python)
    "cpp_line":  "#2e8b8b",   # slate teal    (C++)
    "annot":     "#f0c040",   # gold          (speedup labels)
    "legend_bg": "#14161b",
}

# ── Chart generation ──────────────────────────────────────────────────────────

def generate_chart(
    agent_counts: list[int],
    py_results: dict,
    cpp_results: dict | None,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(_C["bg_fig"])
    ax.set_facecolor(_C["bg_ax"])

    kw = dict(marker="o", linewidth=2.5, markersize=7,
               markeredgewidth=1.2, markeredgecolor="#0b0c0e")

    py_ms = [py_results[n]["avg_ms"] for n in agent_counts]
    ax.plot(agent_counts, py_ms, label="Python", color=_C["py_line"], **kw)

    if cpp_results:
        cpp_ms = [cpp_results[n]["avg_ms"] for n in agent_counts]
        ax.plot(agent_counts, cpp_ms, label="C++ (pybind11)", color=_C["cpp_line"], **kw)

        # Shaded region between curves
        ax.fill_between(agent_counts, py_ms, cpp_ms,
                         alpha=0.08, color=_C["py_line"])

        # Speedup annotations above each C++ data point
        for n, py_t, cpp_t in zip(agent_counts, py_ms, cpp_ms):
            speedup = py_t / cpp_t if cpp_t > 0 else 0
            ax.annotate(
                f"{speedup:.1f}×",
                xy=(n, cpp_t),
                xytext=(0, 10),
                textcoords="offset points",
                color=_C["annot"],
                fontsize=8.5,
                fontweight="bold",
                ha="center",
            )

    ax.set_xlabel("Number of Agents", color=_C["text"], fontsize=12, labelpad=8)
    ax.set_ylabel("Avg Frame Time (ms)", color=_C["text"], fontsize=12, labelpad=8)
    ax.set_title(
        "SimCore Arena — Proximity: Python vs C++ (pybind11)\nAvg frame time per agent count  |  300 frames, fixed Δt",
        color=_C["text"], fontsize=12, pad=14,
    )

    ax.tick_params(colors=_C["text"], which="both", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(_C["border"])

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", color=_C["grid_maj"], linewidth=0.8)
    ax.grid(True, which="minor", color=_C["grid_min"], linewidth=0.4)

    legend = ax.legend(
        facecolor=_C["legend_bg"],
        edgecolor=_C["border"],
        labelcolor=_C["text"],
        fontsize=11,
        framealpha=0.95,
    )

    # Watermark
    fig.text(0.99, 0.01, "SimCore Arena", color=_C["border"],
             fontsize=8, ha="right", va="bottom", style="italic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[Chart] Saved → {output_path}")


# ── Console table ─────────────────────────────────────────────────────────────

def print_table(agent_counts, py_results, cpp_results):
    header = f"{'Agents':>8} | {'Py avg(ms)':>12} | {'Py FPS':>10}"
    if cpp_results:
        header += f" | {'C++ avg(ms)':>12} | {'C++ FPS':>10} | {'Speedup':>8}"
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for n in agent_counts:
        py  = py_results[n]
        row = f"{n:>8} | {py['avg_ms']:>12.3f} | {py['fps']:>10.1f}"
        if cpp_results:
            cpp = cpp_results[n]
            spd = py["avg_ms"] / cpp["avg_ms"] if cpp["avg_ms"] > 0 else 0
            row += f" | {cpp['avg_ms']:>12.3f} | {cpp['fps']:>10.1f} | {spd:>7.2f}×"
        print(row)
    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SimCore Arena benchmark")
    p.add_argument("--agents",  nargs="+", type=int, default=[10, 20, 30, 45, 60],
                   help="Agent counts to test (default: 10 20 30 45 60)")
    p.add_argument("--frames",  type=int, default=300,
                   help="Frames per configuration (default: 300)")
    p.add_argument("--output",  default="benchmark.png",
                   help="Output chart path (default: benchmark.png)")
    p.add_argument("--no-cpp",  action="store_true",
                   help="Skip C++ benchmark even if module is available")
    return p.parse_args()


def main():
    args = parse_args()

    # Try to load C++ module
    cpp_mod = None
    if not args.no_cpp:
        try:
            import proximity_cpp
            cpp_mod = proximity_cpp
            print("[Benchmark] C++ module loaded.")
        except ImportError:
            print("[Benchmark] C++ module not found — running Python only.")

    print(f"\n{'='*60}")
    print(f"  SimCore Arena — Performance Benchmark")
    print(f"  Agents: {args.agents}  |  Frames: {args.frames}")
    print(f"{'='*60}\n")

    print("── Python proximity ──────────────────────────────────────────")
    py_results = run_benchmark(args.agents, args.frames, use_cpp=False)

    cpp_results = None
    if cpp_mod:
        print("\n── C++ proximity ─────────────────────────────────────────────")
        cpp_results = run_benchmark(args.agents, args.frames,
                                    use_cpp=True, cpp_mod=cpp_mod)

    print_table(args.agents, py_results, cpp_results)

    output = os.path.abspath(args.output)
    generate_chart(args.agents, py_results, cpp_results, output)
    print(f"\n[Done] Results chart → {output}")


if __name__ == "__main__":
    main()
