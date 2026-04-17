/*
 * proximity.cpp
 *
 * C++ implementation of O(n²) agent-to-agent proximity detection.
 * Exposed to Python via pybind11.
 *
 * Build:
 *   cd cpp_module && python setup.py build_ext --inplace
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // automatic list / vector conversion

#include <cmath>
#include <vector>
#include <utility>   // std::pair

namespace py = pybind11;

/**
 * find_neighbors
 *
 * Given parallel arrays of X and Y coordinates (one entry per agent),
 * return every pair (i, j) with i < j whose Euclidean distance is
 * strictly less than *radius*.
 *
 * @param xs     X coordinates of all agents
 * @param ys     Y coordinates of all agents
 * @param radius Search radius in world units
 * @return       Vector of (index_i, index_j) pairs
 */
std::vector<std::pair<int, int>>
find_neighbors(const std::vector<double>& xs,
               const std::vector<double>& ys,
               double radius)
{
    std::vector<std::pair<int, int>> result;
    const int n = static_cast<int>(xs.size());
    const double r2 = radius * radius;   // compare squared distances — avoids sqrt

    result.reserve(n * 4);              // rough pre-allocation

    for (int i = 0; i < n; ++i) {
        const double xi = xs[i];
        const double yi = ys[i];
        for (int j = i + 1; j < n; ++j) {
            const double dx = xi - xs[j];
            const double dy = yi - ys[j];
            if (dx * dx + dy * dy < r2) {
                result.emplace_back(i, j);
            }
        }
    }

    return result;
}

/**
 * compute_distances
 *
 * Benchmark helper: compute the full n×n distance matrix.
 * Returns a flat vector of n² doubles (row-major).
 */
std::vector<double>
compute_distances(const std::vector<double>& xs,
                  const std::vector<double>& ys)
{
    const int n = static_cast<int>(xs.size());
    std::vector<double> out(static_cast<size_t>(n) * n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                const double dx = xs[i] - xs[j];
                const double dy = ys[i] - ys[j];
                out[static_cast<size_t>(i) * n + j] = std::sqrt(dx * dx + dy * dy);
            }
        }
    }

    return out;
}

// ── pybind11 module definition ─────────────────────────────────────────────────

PYBIND11_MODULE(proximity_cpp, m) {
    m.doc() = "C++ accelerated proximity detection for SimCore Arena";

    m.def(
        "find_neighbors",
        &find_neighbors,
        py::arg("xs"),
        py::arg("ys"),
        py::arg("radius"),
        R"doc(
Find all agent pairs within *radius* world units of each other.

Parameters
----------
xs     : list[float]  — X coordinates (one per agent, by index)
ys     : list[float]  — Y coordinates (one per agent, by index)
radius : float        — search radius

Returns
-------
list[tuple[int, int]] — (i, j) index pairs with i < j
        )doc"
    );

    m.def(
        "compute_distances",
        &compute_distances,
        py::arg("xs"),
        py::arg("ys"),
        R"doc(
Compute the full n×n pairwise distance matrix.

Returns a flat list of n² floats in row-major order.
        )doc"
    );
}
