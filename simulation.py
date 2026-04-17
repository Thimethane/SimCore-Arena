"""
Simulation system — owns all agents and drives the per-frame update loop.
"""

import random

from panda3d.core import Vec3, Point3

from agents.agent import Agent
from systems.proximity import ProximitySystemPython, ProximitySystemCpp, PROXIMITY_RADIUS

ARENA_HALF = 24.0


class SimulationSystem:
    """
    Manages agent creation, the proximity back-end selection, and
    the per-frame update that advances every agent by *dt* seconds.
    """

    def __init__(self, num_agents: int = 30, mode: str = "python") -> None:
        """
        Parameters
        ----------
        num_agents : int   — how many agents to spawn
        mode       : str   — "python" | "cpp"
        """
        self.mode = mode

        # Shared seek target — all agents in SEEK state move toward this point
        self.seek_target = Point3(0.0, 0.0, 0.0)

        # Spawn agents at random positions inside the arena
        self.agents: list[Agent] = [
            Agent(
                agent_id=i,
                position=Vec3(
                    random.uniform(-ARENA_HALF * 0.8, ARENA_HALF * 0.8),
                    random.uniform(-ARENA_HALF * 0.8, ARENA_HALF * 0.8),
                    0.0,
                ),
                target=self.seek_target,
            )
            for i in range(num_agents)
        ]

        # Select proximity back-end
        if mode == "cpp":
            self._proximity = ProximitySystemCpp()
            if not self._proximity.cpp_available:
                print("[SimulationSystem] Falling back to Python proximity.")
        else:
            self._proximity = ProximitySystemPython()

        print(f"[SimulationSystem] {num_agents} agents | mode={mode}")

    # ── Per-frame update ───────────────────────────────────────────────────────

    def update(self, dt: float) -> None:
        """Advance all agents by *dt* seconds."""
        # 1. Compute neighbours for every agent
        neighbor_map = self._proximity.get_neighbors(self.agents, PROXIMITY_RADIUS)

        # 2. Update each agent with its neighbour list
        for agent in self.agents:
            agent.update(dt, neighbor_map[agent.agent_id])

    # ── Utility ────────────────────────────────────────────────────────────────

    def set_seek_target(self, x: float, y: float) -> None:
        """Move the global seek target (e.g. on mouse click)."""
        self.seek_target.x = x
        self.seek_target.y = y
        # Propagate the new target reference to every agent
        for agent in self.agents:
            agent.target = self.seek_target

    @property
    def agent_count(self) -> int:
        return len(self.agents)
