"""
Agent module for SimCore Arena.

Each agent maintains a position, velocity, and one of three behavioral states:
  - SEEK  : move toward a designated target point
  - AVOID : steer away from agents that are too close
  - IDLE  : wander randomly around the arena
"""

import math
import random
from enum import Enum, auto

from panda3d.core import Vec3, Point3


# ── State definitions ──────────────────────────────────────────────────────────

class AgentState(Enum):
    SEEK  = auto()
    AVOID = auto()
    IDLE  = auto()


# ── Color map (R, G, B) — SimCore Arena palette ────────────────────────────────
# SEEK  → amber-orange  (#e8820c)
# AVOID → deep crimson  (#c0392b)
# IDLE  → slate teal    (#2e8b8b)

STATE_COLORS = {
    AgentState.SEEK:  (0.91, 0.51, 0.047),  # amber-orange
    AgentState.AVOID: (0.75, 0.22, 0.17),   # deep crimson
    AgentState.IDLE:  (0.18, 0.545, 0.545), # slate teal
}

# ── Tunable parameters ─────────────────────────────────────────────────────────

AVOID_RADIUS   = 4.0   # units — if another agent is closer than this, switch to AVOID
SEEK_RADIUS    = 12.0  # units — if the target is within this radius, switch to SEEK
MAX_SPEED      = 5.0   # units / second
IDLE_SPEED     = 2.0   # units / second
WANDER_CHANGE  = 0.02  # probability of picking a new wander direction each frame
ARENA_HALF     = 24.0  # agents are kept inside ±ARENA_HALF on X and Y


class Agent:
    """A single simulation agent."""

    def __init__(self, agent_id: int, position: Vec3, target: Point3):
        self.agent_id = agent_id

        # Spatial state
        self.position = Vec3(position)
        self.velocity = Vec3(
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            0.0,
        )
        self._normalize_velocity(IDLE_SPEED)

        # Behavioural state
        self.state  = AgentState.IDLE
        self.target = target          # global seek target (may be updated externally)

        # Wander heading (used in IDLE)
        self._wander_angle = random.uniform(0, 2 * math.pi)

        # Panda3D scene node — set by the renderer after creation
        self.node_path = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, dt: float, neighbors: list["Agent"]) -> None:
        """Advance the agent by *dt* seconds given a list of nearby agents."""
        self._decide_state(neighbors)
        self._apply_behavior(dt, neighbors)
        self._clamp_to_arena()

        # Sync scene-graph node if it exists
        if self.node_path is not None:
            self.node_path.set_pos(self.position)

    def color(self) -> tuple[float, float, float]:
        """Return the RGB color that represents the current state."""
        return STATE_COLORS[self.state]

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _decide_state(self, neighbors: list["Agent"]) -> None:
        """Choose AVOID > SEEK > IDLE based on proximity and target distance."""
        # AVOID takes priority: any neighbor inside the avoidance radius
        for other in neighbors:
            if self._distance(other.position) < AVOID_RADIUS:
                self.state = AgentState.AVOID
                return

        # SEEK if the target is close enough to be interesting
        target_dist = math.sqrt(
            (self.position.x - self.target.x) ** 2 +
            (self.position.y - self.target.y) ** 2
        )
        if target_dist < SEEK_RADIUS:
            self.state = AgentState.SEEK
            return

        self.state = AgentState.IDLE

    def _apply_behavior(self, dt: float, neighbors: list["Agent"]) -> None:
        if self.state == AgentState.SEEK:
            self._behavior_seek(dt)
        elif self.state == AgentState.AVOID:
            self._behavior_avoid(dt, neighbors)
        else:
            self._behavior_idle(dt)

    def _behavior_seek(self, dt: float) -> None:
        """Steer directly toward the target."""
        dx = self.target.x - self.position.x
        dy = self.target.y - self.position.y
        length = math.hypot(dx, dy)
        if length < 0.01:
            return
        self.velocity = Vec3(dx / length * MAX_SPEED, dy / length * MAX_SPEED, 0.0)
        self.position += self.velocity * dt

    def _behavior_avoid(self, dt: float, neighbors: list["Agent"]) -> None:
        """Compute a repulsion vector away from all nearby agents."""
        repulse_x = 0.0
        repulse_y = 0.0
        count = 0
        for other in neighbors:
            dist = self._distance(other.position)
            if 0.01 < dist < AVOID_RADIUS:
                # Weight repulsion inversely by distance
                weight = (AVOID_RADIUS - dist) / AVOID_RADIUS
                repulse_x += (self.position.x - other.position.x) / dist * weight
                repulse_y += (self.position.y - other.position.y) / dist * weight
                count += 1

        if count == 0:
            self._behavior_idle(dt)
            return

        length = math.hypot(repulse_x, repulse_y)
        if length > 0.01:
            self.velocity = Vec3(
                repulse_x / length * MAX_SPEED,
                repulse_y / length * MAX_SPEED,
                0.0,
            )
        self.position += self.velocity * dt

    def _behavior_idle(self, dt: float) -> None:
        """Wander by slowly drifting the heading angle."""
        if random.random() < WANDER_CHANGE:
            self._wander_angle += random.uniform(-0.8, 0.8)

        self.velocity = Vec3(
            math.cos(self._wander_angle) * IDLE_SPEED,
            math.sin(self._wander_angle) * IDLE_SPEED,
            0.0,
        )
        self.position += self.velocity * dt

    def _clamp_to_arena(self) -> None:
        """Bounce the agent off the arena boundaries."""
        if self.position.x >  ARENA_HALF:
            self.position.x  =  ARENA_HALF
            self.velocity.x  = -abs(self.velocity.x)
            self._wander_angle = math.pi - self._wander_angle
        if self.position.x < -ARENA_HALF:
            self.position.x  = -ARENA_HALF
            self.velocity.x  =  abs(self.velocity.x)
            self._wander_angle = math.pi - self._wander_angle
        if self.position.y >  ARENA_HALF:
            self.position.y  =  ARENA_HALF
            self.velocity.y  = -abs(self.velocity.y)
            self._wander_angle = -self._wander_angle
        if self.position.y < -ARENA_HALF:
            self.position.y  = -ARENA_HALF
            self.velocity.y  =  abs(self.velocity.y)
            self._wander_angle = -self._wander_angle

    def _distance(self, other_pos: Vec3) -> float:
        dx = self.position.x - other_pos.x
        dy = self.position.y - other_pos.y
        return math.sqrt(dx * dx + dy * dy)

    def _normalize_velocity(self, speed: float) -> None:
        length = math.hypot(self.velocity.x, self.velocity.y)
        if length > 0.01:
            self.velocity.x = self.velocity.x / length * speed
            self.velocity.y = self.velocity.y / length * speed
