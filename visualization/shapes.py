"""Wireframe meshes (unit-radius vertex sets + edge lists) for the 3D-looking objects.

Built once at import time and made read-only; the renderer rotates and projects
them every frame.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray  # (N, 3), fits inside the unit sphere
    edges: np.ndarray  # (M, 2) int indices

    def __post_init__(self) -> None:
        self.vertices.setflags(write=False)
        self.edges.setflags(write=False)


def _normalise(v: np.ndarray) -> np.ndarray:
    return v / np.max(np.linalg.norm(v, axis=1))


def _edges_by_length(v: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Connect every pair of vertices at the minimal pairwise distance (works for the
    regular solids where all edges have equal length)."""
    pairs = list(itertools.combinations(range(len(v)), 2))
    dist = np.array([np.linalg.norm(v[a] - v[b]) for a, b in pairs])
    return np.array([p for p, d in zip(pairs, dist) if d < dist.min() + tol])


def _cube() -> Mesh:
    v = _normalise(np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float))
    return Mesh(v, _edges_by_length(v))


def _tetrahedron() -> Mesh:
    v = _normalise(np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float))
    return Mesh(v, _edges_by_length(v))


def _octahedron() -> Mesh:
    v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    return Mesh(v, _edges_by_length(v))


def _icosahedron() -> Mesh:
    phi = (1 + 5**0.5) / 2
    v = []
    for a, b in itertools.product((-1, 1), repeat=2):
        v += [[0, a, b * phi], [a, b * phi, 0], [b * phi, 0, a]]
    v = _normalise(np.array(v, dtype=float))
    return Mesh(v, _edges_by_length(v))


def _bipyramid(sides: int = 5) -> Mesh:
    ring = [[np.cos(a), 0.0, np.sin(a)] for a in np.linspace(0, 2 * np.pi, sides, endpoint=False)]
    v = np.array(ring + [[0, 1.25, 0], [0, -1.25, 0]], dtype=float)
    v = _normalise(v)
    edges = [(i, (i + 1) % sides) for i in range(sides)]
    edges += [(i, sides) for i in range(sides)] + [(i, sides + 1) for i in range(sides)]
    return Mesh(v, np.array(edges))


def _torus(major: int = 14, minor: int = 6, r_major: float = 0.72, r_minor: float = 0.28) -> Mesh:
    pts, edges = [], []
    for i in range(major):
        u = 2 * np.pi * i / major
        for j in range(minor):
            w = 2 * np.pi * j / minor
            pts.append([(r_major + r_minor * np.cos(w)) * np.cos(u), r_minor * np.sin(w), (r_major + r_minor * np.cos(w)) * np.sin(u)])
            idx = i * minor + j
            edges.append((idx, i * minor + (j + 1) % minor))  # minor ring
            edges.append((idx, ((i + 1) % major) * minor + j))  # along the tube
    return Mesh(np.array(pts), np.array(edges))


def _sphere(rings: int = 6, segments: int = 12) -> Mesh:
    pts, edges = [], []
    for i in range(1, rings):  # latitude rings, poles excluded
        lat = np.pi * i / rings - np.pi / 2
        for j in range(segments):
            lon = 2 * np.pi * j / segments
            pts.append([np.cos(lat) * np.cos(lon), np.sin(lat), np.cos(lat) * np.sin(lon)])
            idx = (i - 1) * segments + j
            edges.append((idx, (i - 1) * segments + (j + 1) % segments))
            if i > 1:
                edges.append((idx, (i - 2) * segments + j))  # meridians
    return Mesh(np.array(pts), np.array(edges))


MESHES: dict[str, Mesh] = {
    "cube": _cube(),
    "tetrahedron": _tetrahedron(),
    "octahedron": _octahedron(),
    "icosahedron": _icosahedron(),
    "bipyramid": _bipyramid(),
    "torus": _torus(),
    "sphere": _sphere(),
}
