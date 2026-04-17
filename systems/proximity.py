"""
Proximity system — finds neighbours for every agent.

Two back-ends are provided:

  ProximitySystemPython  — pure Python O(n²) scan
  ProximitySystemCpp     — same algorithm in C++ via pybind11

Both expose an identical interface:
    get_neighbors(agents, radius) -> dict[int, list[Agent]]
"""

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.agent import Agent

# Detection radius used by the simulation (slightly larger than AVOID_RADIUS
# so the AI has a small look-ahead window before a collision).
PROXIMITY_RADIUS = 6.0


class ProximitySystemPython:
    """
    Pure-Python O(n²) proximity scan.

    For each agent we iterate over all other agents and collect those
    whose Euclidean distance is within *radius* units.
    """

    def get_neighbors(
        self,
        agents: list["Agent"],
        radius: float = PROXIMITY_RADIUS,
    ) -> dict[int, list["Agent"]]:
        """
        Return a mapping  agent_id -> [neighbouring Agent objects].

        Parameters
        ----------
        agents : list of Agent
        radius : float  — search radius in world units

        Returns
        -------
        dict[int, list[Agent]]
        """
        neighbors: dict[int, list] = {a.agent_id: [] for a in agents}

        n = len(agents)
        for i in range(n):
            ai = agents[i]
            xi, yi = ai.position.x, ai.position.y
            for j in range(i + 1, n):
                aj = agents[j]
                dx = xi - aj.position.x
                dy = yi - aj.position.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < radius:
                    neighbors[ai.agent_id].append(aj)
                    neighbors[aj.agent_id].append(ai)

        return neighbors


class ProximitySystemCpp:
    """
    C++ accelerated proximity scan (same O(n²) algorithm, faster arithmetic).

    Falls back to the Python implementation if the compiled module is not
    available so the simulation always runs.
    """

    def __init__(self) -> None:
        self._cpp_available = False
        try:
            import proximity_cpp  # noqa: F401  (built by setup.py)
            self._mod = proximity_cpp
            self._cpp_available = True
            print("[ProximitySystemCpp] C++ module loaded successfully.")
        except ImportError:
            print("[ProximitySystemCpp] WARNING: C++ module not found — "
                  "falling back to Python implementation.")
            self._fallback = ProximitySystemPython()

    @property
    def cpp_available(self) -> bool:
        return self._cpp_available

    def get_neighbors(
        self,
        agents: list["Agent"],
        radius: float = PROXIMITY_RADIUS,
    ) -> dict[int, list["Agent"]]:
        if not self._cpp_available:
            return self._fallback.get_neighbors(agents, radius)

        # Pack positions into flat lists for the C++ call
        ids = [a.agent_id for a in agents]
        xs  = [a.position.x for a in agents]
        ys  = [a.position.y for a in agents]

        # C++ returns: list of (id_i, id_j) pairs within radius
        pairs = self._mod.find_neighbors(xs, ys, radius)

        # Build an id->Agent lookup and reconstruct the neighbour dict
        id_to_agent = {a.agent_id: a for a in agents}
        neighbors: dict[int, list] = {aid: [] for aid in ids}
        for id_i, id_j in pairs:
            neighbors[id_i].append(id_to_agent[id_j])
            neighbors[id_j].append(id_to_agent[id_i])

        return neighbors
