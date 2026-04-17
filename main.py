"""
main.py — SimCore Arena
==============================

Entry point for the real-time 3D simulation.

Usage
-----
    python main.py                   # 30 agents, Python proximity
    python main.py --mode cpp        # 30 agents, C++ proximity
    python main.py --agents 60       # 60 agents
    python main.py --agents 60 --mode cpp

Controls
--------
    Arrow keys / WASD  — pan camera
    Mouse wheel        — zoom
    Left-click         — move seek target to clicked position
    ESC                — quit
"""

import sys
import argparse
import time

from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (
    AmbientLight, DirectionalLight,
    Vec4, Vec3, Point3, LColor,
    TextNode, CardMaker, NodePath,
    CollisionTraverser, CollisionNode, CollisionRay, CollisionHandlerQueue,
    GeomNode,
)
from direct.task import Task

from systems.simulation import SimulationSystem
from agents.agent import AgentState, STATE_COLORS, ARENA_HALF

# ── Visual constants ───────────────────────────────────────────────────────────

AGENT_SCALE   = 0.5    # visual sphere scale
TARGET_SCALE  = 0.8    # seek-target marker scale
GROUND_SIZE   = ARENA_HALF * 2 + 4

# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SimCore Arena")
    p.add_argument("--agents", type=int, default=30,
                   help="Number of agents (default: 30)")
    p.add_argument("--mode", choices=["python", "cpp"], default="python",
                   help="Proximity back-end (default: python)")
    return p.parse_args()


# ── Application ────────────────────────────────────────────────────────────────

class SimulationApp(ShowBase):
    """Panda3D application driving the AI simulation."""

    def __init__(self, num_agents: int, mode: str) -> None:
        super().__init__()

        self.mode = mode
        self.disableMouse()          # we control the camera manually

        # ── Scene ──────────────────────────────────────────────────────────────
        self._setup_camera()
        self._setup_lighting()
        self._setup_ground()
        self._setup_target_marker()

        # ── Simulation ─────────────────────────────────────────────────────────
        self.sim = SimulationSystem(num_agents=num_agents, mode=mode)
        self._agent_nodes: dict[int, NodePath] = {}
        self._spawn_agent_visuals()

        # ── HUD ────────────────────────────────────────────────────────────────
        self._setup_hud()

        # ── Input ──────────────────────────────────────────────────────────────
        self._setup_input()

        # ── Collision picker (click-to-set-target) ─────────────────────────────
        self._setup_picker()

        # ── Timing ─────────────────────────────────────────────────────────────
        self._last_time  = time.perf_counter()
        self._frame_times: list[float] = []

        # ── Task ───────────────────────────────────────────────────────────────
        self.taskMgr.add(self._simulation_task, "SimulationTask")
        self.taskMgr.add(self._hud_task,        "HUDTask")

    # ── Camera ────────────────────────────────────────────────────────────────

    def _setup_camera(self) -> None:
        """Top-down orthographic-style camera with pan support."""
        self.camera.set_pos(0, -60, 55)
        self.camera.look_at(0, 0, 0)

    # ── Lighting ──────────────────────────────────────────────────────────────

    def _setup_lighting(self) -> None:
        # Dim cool ambient keeps the dark charcoal ground readable
        ambient = AmbientLight("ambient")
        ambient.set_color(Vec4(0.28, 0.28, 0.30, 1))
        self.render.set_light(self.render.attach_new_node(ambient))

        # Warm-tinted key light from upper-left
        sun = DirectionalLight("sun")
        sun.set_color(Vec4(1.0, 0.88, 0.72, 1))
        sun_np = self.render.attach_new_node(sun)
        sun_np.set_hpr(-40, -55, 0)
        self.render.set_light(sun_np)

    # ── Ground ────────────────────────────────────────────────────────────────

    def _setup_ground(self) -> None:
        cm = CardMaker("ground")
        cm.set_frame(-ARENA_HALF - 2, ARENA_HALF + 2,
                     -ARENA_HALF - 2, ARENA_HALF + 2)
        ground = self.render.attach_new_node(cm.generate())
        ground.set_p(-90)
        # SimCore Arena ground: near-black charcoal (#0b0c0e ≈ 0.043, 0.047, 0.055)
        ground.set_color(0.043, 0.047, 0.055, 1)

        # Arena boundary lines
        self._draw_boundary()

    def _draw_boundary(self) -> None:
        """Draw a visible boundary around the arena using thin flat boxes."""
        thickness = 0.15
        h = ARENA_HALF
        walls = [
            # (x, y, sx, sy)  — position and half-extents for each wall segment
            ( h,  0, thickness, h),
            (-h,  0, thickness, h),
            ( 0,  h, h, thickness),
            ( 0, -h, h, thickness),
        ]
        for i, (x, y, sx, sy) in enumerate(walls):
            cm = CardMaker(f"wall_{i}")
            cm.set_frame(-sx, sx, -sy, sy)
            w = self.render.attach_new_node(cm.generate())
            w.set_pos(x, y, 0.02)
            w.set_p(-90)
            # SimCore Arena border: muted steel (#2e3340 ≈ 0.18, 0.20, 0.25)
            w.set_color(0.18, 0.20, 0.25, 1)

    # ── Seek target marker ────────────────────────────────────────────────────

    def _setup_target_marker(self) -> None:
        self._target_np = self.loader.load_model("models/misc/sphere")
        self._target_np.set_scale(TARGET_SCALE)
        self._target_np.set_color(0.941, 0.753, 0.251, 1)  # SimCore gold (#f0c040)
        self._target_np.set_pos(0, 0, 0.5)
        self._target_np.reparent_to(self.render)

    # ── Agent visuals ─────────────────────────────────────────────────────────

    def _spawn_agent_visuals(self) -> None:
        for agent in self.sim.agents:
            np = self.loader.load_model("models/misc/sphere")
            np.set_scale(AGENT_SCALE)
            r, g, b = agent.color()
            np.set_color(r, g, b, 1)
            np.set_pos(agent.position)
            np.reparent_to(self.render)
            self._agent_nodes[agent.agent_id] = np
            agent.node_path = np

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _setup_hud(self) -> None:
        # SimCore Arena HUD — warm off-white text (#e8dcc8) on dark shadow
        style = dict(
            fg=(0.91, 0.863, 0.784, 1),   # #e8dcc8 warm off-white
            shadow=(0, 0, 0, 0.75),
            scale=0.06,
            align=TextNode.A_left,
            mayChange=True,
        )
        self._fps_text   = OnscreenText(text="FPS: --",     pos=(-1.3,  0.92), **style)
        self._frame_text = OnscreenText(text="Frame: --ms", pos=(-1.3,  0.84), **style)
        self._mode_text  = OnscreenText(
            text=f"Mode: {self.mode.upper()}  |  Agents: {self.sim.agent_count}",
            pos=(-1.3, 0.76), **style
        )
        self._state_legend()

    def _state_legend(self) -> None:
        labels = [
            ("■ SEEK",  STATE_COLORS[AgentState.SEEK],  0.68),
            ("■ AVOID", STATE_COLORS[AgentState.AVOID], 0.60),
            ("■ IDLE",  STATE_COLORS[AgentState.IDLE],  0.52),
        ]
        for text, (r, g, b), y in labels:
            OnscreenText(
                text=text, pos=(-1.3, y),
                fg=(r, g, b, 1), shadow=(0, 0, 0, 0.5),
                scale=0.055, align=TextNode.A_left,
            )

    def _hud_task(self, task: Task) -> int:
        if self._frame_times:
            avg_ms = sum(self._frame_times[-60:]) / len(self._frame_times[-60:]) * 1000
            fps    = 1000.0 / avg_ms if avg_ms > 0 else 0
            self._fps_text.setText(f"FPS: {fps:.1f}")
            self._frame_text.setText(f"Frame: {avg_ms:.2f} ms")
        return Task.cont

    # ── Input ─────────────────────────────────────────────────────────────────

    def _setup_input(self) -> None:
        self.accept("escape", sys.exit)

        # Camera pan
        speed = 0.5
        self.taskMgr.add(self._camera_task, "CameraTask")
        self._cam_keys = {k: False for k in
                          ["arrow_up", "arrow_down", "arrow_left", "arrow_right",
                           "w", "s", "a", "d", "wheel_up", "wheel_down"]}
        for k in self._cam_keys:
            self.accept(k,        self._set_key, [k, True])
            self.accept(k + "-up", self._set_key, [k, False])

        self.accept("wheel_up",   self._zoom, [3])
        self.accept("wheel_down", self._zoom, [-3])

        # Click to set seek target
        self.accept("mouse1", self._on_click)

    def _set_key(self, key: str, val: bool) -> None:
        self._cam_keys[key] = val

    def _zoom(self, delta: float) -> None:
        self.camera.set_y(self.camera, delta)

    def _camera_task(self, task: Task) -> int:
        spd = 0.3
        if self._cam_keys["arrow_up"]    or self._cam_keys["w"]: self.camera.set_y(self.camera,  spd)
        if self._cam_keys["arrow_down"]  or self._cam_keys["s"]: self.camera.set_y(self.camera, -spd)
        if self._cam_keys["arrow_left"]  or self._cam_keys["a"]: self.camera.set_x(self.camera, -spd)
        if self._cam_keys["arrow_right"] or self._cam_keys["d"]: self.camera.set_x(self.camera,  spd)
        return Task.cont

    # ── Click-to-target picker ────────────────────────────────────────────────

    def _setup_picker(self) -> None:
        self._picker = CollisionTraverser()
        self._pick_queue = CollisionHandlerQueue()
        pick_node = CollisionNode("mouseRay")
        pick_node.set_from_collide_mask(0x1)
        self._pick_ray = CollisionRay()
        pick_node.add_solid(self._pick_ray)
        pick_np = self.camera.attach_new_node(pick_node)
        self._picker.add_collider(pick_np, self._pick_queue)

    def _on_click(self) -> None:
        if not self.mouseWatcherNode.has_mouse():
            return
        mpos = self.mouseWatcherNode.get_mouse()
        self._pick_ray.set_from_lens(self.camNode, mpos.x, mpos.y)
        # Simple ground-plane intersection (z=0 plane)
        pos = self._ray_ground_intersection()
        if pos is not None:
            self.sim.set_seek_target(pos.x, pos.y)
            self._target_np.set_pos(pos.x, pos.y, 0.5)

    def _ray_ground_intersection(self) -> Point3 | None:
        """Intersect camera ray with z=0 ground plane."""
        if not self.mouseWatcherNode.has_mouse():
            return None
        mpos = self.mouseWatcherNode.get_mouse()
        near = Point3()
        far  = Point3()
        self.camLens.extrude(mpos, near, far)
        # Transform to world space
        near = self.render.get_relative_point(self.camera, near)
        far  = self.render.get_relative_point(self.camera, far)
        # Ray-plane (z=0) intersection
        dz = far.z - near.z
        if abs(dz) < 1e-6:
            return None
        t = -near.z / dz
        x = near.x + t * (far.x - near.x)
        y = near.y + t * (far.y - near.y)
        return Point3(x, y, 0)

    # ── Main simulation task ───────────────────────────────────────────────────

    def _simulation_task(self, task: Task) -> int:
        now = time.perf_counter()
        dt  = now - self._last_time
        self._last_time = now
        dt = min(dt, 0.05)   # cap to avoid spiral-of-death on lag spikes

        # Advance simulation
        self.sim.update(dt)

        # Sync colours (state may have changed)
        for agent in self.sim.agents:
            r, g, b = agent.color()
            self._agent_nodes[agent.agent_id].set_color(r, g, b, 1)

        self._frame_times.append(dt)
        return Task.cont


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    app  = SimulationApp(num_agents=args.agents, mode=args.mode)
    app.run()
