"""
generate_demo.py — Headless demo GIF generator
================================================

Runs the simulation for N frames, renders each frame using matplotlib
(no display server needed), and stitches the frames into demo.gif.

Usage
-----
    python generate_demo.py
    python generate_demo.py --agents 40 --frames 120 --fps 20 --output demo.gif
"""

import argparse
import math
import random
import sys
import os
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from PIL import Image

# ── Reuse the lightweight simulation core from benchmark.py ──────────────────

sys.path.insert(0, os.path.dirname(__file__))

# --- inline the minimal sim (avoids Panda3D import in headless context) ------

import math, random

AVOID_RADIUS     = 4.0
SEEK_RADIUS      = 12.0
MAX_SPEED        = 5.0
IDLE_SPEED       = 2.0
WANDER_CHANGE    = 0.025
ARENA_HALF       = 24.0
PROXIMITY_RADIUS = 6.0

STATE_COLORS = {
    "SEEK":  "#e8820c",   # amber-orange
    "AVOID": "#c0392b",   # deep crimson
    "IDLE":  "#2e8b8b",   # slate teal
}
STATE_EDGE = {
    "SEEK":  "#c46800",
    "AVOID": "#922b21",
    "IDLE":  "#1f6b6b",
}


class Vec2:
    __slots__ = ("x", "y")
    def __init__(self, x=0., y=0.): self.x, self.y = x, y
    def __repr__(self): return f"Vec2({self.x:.2f}, {self.y:.2f})"


class Agent:
    def __init__(self, aid, x, y):
        self.aid   = aid
        self.pos   = Vec2(x, y)
        self.vel   = Vec2(random.uniform(-1, 1), random.uniform(-1, 1))
        self._norm(IDLE_SPEED)
        self.state = "IDLE"
        self.target = Vec2(0., 0.)
        self._wangle = random.uniform(0, 6.2832)

    def update(self, dt, neighbors):
        self._decide(neighbors)
        if   self.state == "SEEK":  self._seek(dt)
        elif self.state == "AVOID": self._avoid(dt, neighbors)
        else:                       self._idle(dt)
        self._clamp()

    def _decide(self, nb):
        for n in nb:
            if self._dist(n.pos) < AVOID_RADIUS:
                self.state = "AVOID"; return
        dx = self.target.x - self.pos.x
        dy = self.target.y - self.pos.y
        if math.sqrt(dx*dx + dy*dy) < SEEK_RADIUS:
            self.state = "SEEK"; return
        self.state = "IDLE"

    def _seek(self, dt):
        dx = self.target.x - self.pos.x
        dy = self.target.y - self.pos.y
        ln = math.hypot(dx, dy)
        if ln < 0.01: return
        self.vel.x = dx/ln * MAX_SPEED
        self.vel.y = dy/ln * MAX_SPEED
        self.pos.x += self.vel.x * dt
        self.pos.y += self.vel.y * dt

    def _avoid(self, dt, nb):
        rx = ry = 0.
        for n in nb:
            d = self._dist(n.pos)
            if 0.01 < d < AVOID_RADIUS:
                w = (AVOID_RADIUS - d) / AVOID_RADIUS
                rx += (self.pos.x - n.pos.x) / d * w
                ry += (self.pos.y - n.pos.y) / d * w
        ln = math.hypot(rx, ry)
        if ln > 0.01:
            self.vel.x = rx/ln * MAX_SPEED
            self.vel.y = ry/ln * MAX_SPEED
        self.pos.x += self.vel.x * dt
        self.pos.y += self.vel.y * dt

    def _idle(self, dt):
        if random.random() < WANDER_CHANGE:
            self._wangle += random.uniform(-0.9, 0.9)
        self.vel.x = math.cos(self._wangle) * IDLE_SPEED
        self.vel.y = math.sin(self._wangle) * IDLE_SPEED
        self.pos.x += self.vel.x * dt
        self.pos.y += self.vel.y * dt

    def _clamp(self):
        H = ARENA_HALF
        if self.pos.x >  H: self.pos.x =  H; self.vel.x = -abs(self.vel.x); self._wangle = math.pi - self._wangle
        if self.pos.x < -H: self.pos.x = -H; self.vel.x =  abs(self.vel.x); self._wangle = math.pi - self._wangle
        if self.pos.y >  H: self.pos.y =  H; self.vel.y = -abs(self.vel.y); self._wangle = -self._wangle
        if self.pos.y < -H: self.pos.y = -H; self.vel.y =  abs(self.vel.y); self._wangle = -self._wangle

    def _dist(self, o): return math.sqrt((self.pos.x-o.x)**2 + (self.pos.y-o.y)**2)
    def _norm(self, s):
        ln = math.hypot(self.vel.x, self.vel.y)
        if ln > 0.01: self.vel.x = self.vel.x/ln*s; self.vel.y = self.vel.y/ln*s


def proximity(agents, radius):
    nb = {a.aid: [] for a in agents}
    n  = len(agents)
    r2 = radius * radius
    for i in range(n):
        ai = agents[i]
        for j in range(i+1, n):
            aj = agents[j]
            dx = ai.pos.x - aj.pos.x
            dy = ai.pos.y - aj.pos.y
            if dx*dx + dy*dy < r2:
                nb[ai.aid].append(aj)
                nb[aj.aid].append(ai)
    return nb


# ── Renderer ──────────────────────────────────────────────────────────────────

BG_COLOR    = "#0b0c0e"
GRID_COLOR  = "#1c1f26"
BORDER_COLOR = "#2e3340"
ARROW_ALPHA = 0.55
ARROW_SCALE = 1.4      # velocity arrow length multiplier


def render_frame(agents, target, frame_idx, total_frames, mode_label):
    """Render one simulation frame → return PIL Image."""
    fig, ax = plt.subplots(figsize=(7, 7), dpi=80)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    H = ARENA_HALF

    # ── Grid ─────────────────────────────────────────────────────────────────
    for v in range(-int(H), int(H)+1, 6):
        ax.axhline(v, color=GRID_COLOR, linewidth=0.5, zorder=0)
        ax.axvline(v, color=GRID_COLOR, linewidth=0.5, zorder=0)

    # ── Arena border ─────────────────────────────────────────────────────────
    border = plt.Rectangle((-H, -H), H*2, H*2,
                            linewidth=2, edgecolor=BORDER_COLOR,
                            facecolor="none", zorder=1)
    ax.add_patch(border)

    # ── Seek target ───────────────────────────────────────────────────────────
    ax.plot(target.x, target.y, "*", color="#f0c040",
            markersize=16, zorder=5, markeredgecolor="#b08800", markeredgewidth=0.8)
    # Seek radius ring
    ring = plt.Circle((target.x, target.y), SEEK_RADIUS,
                       facecolor="#f0c04018", fill=True, linewidth=1,
                       edgecolor="#f0c04055", zorder=2)
    ax.add_patch(ring)

    # ── Agents ───────────────────────────────────────────────────────────────
    for agent in agents:
        color = STATE_COLORS[agent.state]
        edge  = STATE_EDGE[agent.state]

        # Body
        circle = plt.Circle((agent.pos.x, agent.pos.y), 0.7,
                             facecolor=color, zorder=4,
                             linewidth=0.8, edgecolor=edge)
        ax.add_patch(circle)

        # Velocity arrow
        spd = math.hypot(agent.vel.x, agent.vel.y)
        if spd > 0.1:
            scale = ARROW_SCALE / spd
            ax.annotate(
                "", 
                xy=(agent.pos.x + agent.vel.x * scale,
                    agent.pos.y + agent.vel.y * scale),
                xytext=(agent.pos.x, agent.pos.y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    alpha=ARROW_ALPHA,
                    lw=1.2,
                    mutation_scale=8,
                ),
                zorder=3,
            )

        # Avoid radius halo (only when avoiding)
        if agent.state == "AVOID":
            halo = plt.Circle((agent.pos.x, agent.pos.y), AVOID_RADIUS,
                               facecolor="#c0392b18", fill=True,
                               linewidth=0.6, edgecolor="#c0392b44", zorder=2)
            ax.add_patch(halo)

    # ── State counts ─────────────────────────────────────────────────────────
    counts = {"SEEK": 0, "AVOID": 0, "IDLE": 0}
    for a in agents:
        counts[a.state] += 1

    # ── Legend / HUD ─────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=STATE_COLORS[s],
                       label=f"{s}  {counts[s]}")
        for s in ("SEEK", "AVOID", "IDLE")
    ]
    leg = ax.legend(handles=legend_patches, loc="upper right",
                    facecolor="#14161b", edgecolor=BORDER_COLOR,
                    labelcolor="#e8dcc8", fontsize=9, framealpha=0.9)

    # Progress bar (drawn in data coordinates along the bottom edge)
    progress = frame_idx / max(total_frames - 1, 1)
    bar_y = -H - 1.2
    ax.plot([-H, H], [bar_y, bar_y], color=GRID_COLOR, linewidth=3,
            solid_capstyle="butt", zorder=6, clip_on=False)
    ax.plot([-H, -H + (H * 2) * progress], [bar_y, bar_y],
            color="#e8820c", linewidth=3, solid_capstyle="butt",
            zorder=7, clip_on=False)

    # Title / info
    ax.set_title(
        f"SimCore Arena  —  {len(agents)} agents  |  {mode_label}  |  "
        f"frame {frame_idx+1}/{total_frames}",
        color="#e8dcc8", fontsize=10, pad=8,
    )

    ax.set_xlim(-H - 1, H + 1)
    ax.set_ylim(-H - 2.5, H + 1)
    ax.set_aspect("equal")
    ax.tick_params(colors="#6a7080", labelsize=7)
    ax.spines[:].set_color(BORDER_COLOR)

    # Bottom-right watermark
    ax.text(H, -H - 2.1, "SimCore Arena",
            color="#2e3340", fontsize=7, ha="right", va="bottom",
            style="italic", zorder=10)

    # ── Capture to PIL ────────────────────────────────────────────────────────
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=80)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── GIF assembly ──────────────────────────────────────────────────────────────

def generate_gif(
    num_agents: int,
    total_frames: int,
    fps: int,
    output_path: str,
    mode: str = "python",
    seed: int = 42,
) -> None:
    random.seed(seed)

    # Try C++ back-end
    cpp_mod = None
    if mode == "cpp":
        try:
            import proximity_cpp
            cpp_mod = proximity_cpp
            print("[GIF] Using C++ proximity.")
        except ImportError:
            print("[GIF] C++ module not found — using Python.")

    # Spawn agents
    agents = [
        Agent(i,
              random.uniform(-ARENA_HALF * 0.75, ARENA_HALF * 0.75),
              random.uniform(-ARENA_HALF * 0.75, ARENA_HALF * 0.75))
        for i in range(num_agents)
    ]
    target = Vec2(0., 0.)
    for a in agents:
        a.target = target

    # Seek target trajectory — slowly orbiting circle
    def get_target_pos(frame):
        angle = (frame / total_frames) * 2 * math.pi * 2   # 2 full orbits
        r = ARENA_HALF * 0.55
        return Vec2(math.cos(angle) * r, math.sin(angle) * r)

    dt = 1.0 / fps
    mode_label = "C++ (pybind11)" if (cpp_mod and mode == "cpp") else "Python"
    frames: list[Image.Image] = []

    print(f"[GIF] Rendering {total_frames} frames ({num_agents} agents) …")
    for f in range(total_frames):
        # Move target
        tp = get_target_pos(f)
        target.x, target.y = tp.x, tp.y

        # Simulation step
        if cpp_mod and mode == "cpp":
            xs = [a.pos.x for a in agents]
            ys = [a.pos.y for a in agents]
            pairs = cpp_mod.find_neighbors(xs, ys, PROXIMITY_RADIUS)
            id_map = {a.aid: a for a in agents}
            nb_map = {a.aid: [] for a in agents}
            for i, j in pairs:
                nb_map[i].append(id_map[j])
                nb_map[j].append(id_map[i])
        else:
            nb_map = proximity(agents, PROXIMITY_RADIUS)

        for agent in agents:
            agent.update(dt, nb_map[agent.aid])

        img = render_frame(agents, target, f, total_frames, mode_label)
        frames.append(img)

        if (f + 1) % 20 == 0 or f == total_frames - 1:
            print(f"  {f+1}/{total_frames} frames rendered")

    # Save GIF
    duration_ms = int(1000 / fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    size_kb = os.path.getsize(output_path) / 1024
    print(f"[GIF] Saved → {output_path}  ({size_kb:.0f} KB, "
          f"{len(frames)} frames @ {fps} fps)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate demo.gif for SimCore Arena")
    p.add_argument("--agents",  type=int, default=35,   help="Agent count (default: 35)")
    p.add_argument("--frames",  type=int, default=90,   help="Total frames (default: 90)")
    p.add_argument("--fps",     type=int, default=18,   help="GIF frame rate (default: 18)")
    p.add_argument("--mode",    choices=["python", "cpp"], default="python")
    p.add_argument("--output",  default="demo.gif",     help="Output path")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_gif(
        num_agents=args.agents,
        total_frames=args.frames,
        fps=args.fps,
        output_path=args.output,
        mode=args.mode,
        seed=args.seed,
    )
