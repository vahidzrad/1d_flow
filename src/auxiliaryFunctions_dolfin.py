#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility routines for legacy FEniCS (dolfin) workflows.
Stripped of any dolfinx–specific calls; everything here imports
from the classic dolfin 2019.1.x API only.

Author: Haifeng Wang  ·  Cleanup: ChatGPT (July-2025)
"""
# from __future__ import annotations

import os
import shutil
import math

import numpy as np
from mpi4py import MPI

# -----------------------------------------------------------------------------
# Legacy FEniCS imports
# -----------------------------------------------------------------------------
from dolfin import (
    Mesh, MeshFunction, FunctionSpace, VectorFunctionSpace, Function,
    TrialFunction, TestFunction, UserExpression, project,
    assemble_system, dx, inner, Constant,
    cells, PETScKrylovSolver
)

# -----------------------------------------------------------------------------
# Simple filesystem helpers
# -----------------------------------------------------------------------------


def remove_recreate_folder(folder_name: str) -> None:
    """Delete *folder_name* if it exists, then recreate it."""
    folder_path = os.path.abspath(folder_name)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_name}' removed.")
    os.makedirs(folder_path, exist_ok=True)
    print(f"Folder '{folder_name}' created.")


def create_folder_if_not_exists(folder_name: str) -> None:
    os.makedirs(folder_name, exist_ok=True)
    print(f"Folder '{folder_name}' ensured.")

# -----------------------------------------------------------------------------
# Lightweight checkpointing (NumPy .npz, MPI-aware)
# -----------------------------------------------------------------------------


def _vec_to_array(f) -> np.ndarray:
    return f.vector().get_local()


def _array_to_vec(f, arr: np.ndarray) -> None:
    f.vector().set_local(arr)
    f.vector().apply("insert")


def save_state(u: Function, filename: str = "./stateBackup/state.npz") -> None:
    np.savez_compressed(filename, u=_vec_to_array(u))


def load_state(u: Function, filename: str = "./stateBackup/state.npz") -> None:
    data = np.load(filename)
    _array_to_vec(u, data["u"])


# MPI-splitting: each rank stores its local vector slice
_comm = MPI.COMM_WORLD


def save_state_mpi(
    step: int,
    u: Function,
    name_tpl: str = "./stateBackup/state_rank{:d}.npz",
) -> None:
    local = _vec_to_array(u)
    np.savez_compressed(name_tpl.format(_comm.rank), step=step, u=local)


def load_state_mpi(
    u: Function,
    name_tpl: str = "./stateBackup/state_rank{:d}.npz",
) -> int:
    fname = name_tpl.format(_comm.rank)
    if not os.path.exists(fname):
        print(f"Rank {_comm.rank}: no checkpoint; starting fresh.")
        step = 0
    else:
        data = np.load(fname)
        step = int(data["step"][()])
        _array_to_vec(u, data["u"])
    # ensure all ranks share the same step
    step = _comm.bcast(step, root=0)
    return step

# -----------------------------------------------------------------------------
# Simple analytic initial-condition helpers
# -----------------------------------------------------------------------------


tolCoord = 1e-5


class GaussianHill(UserExpression):
    """1-D Gaussian in the *y*-direction centred at y=0.3 mm."""

    def eval(self, value, x):  # noqa: D401
        x0, L = 0.3, 0.03  # centre & width (10× smaller)
        B = 95.0 * 1.35e-12
        value[0] = B * math.exp(-((x[1] - x0) / L) ** 2)

    def value_shape(self):
        return ()


class DeltaPulse(UserExpression):
    """Approximate delta-pulse along *y* using a narrow Gaussian."""

    def __init__(self, rl: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self.rl = rl

    def eval(self, value, x):
        pO2C = 1.35e-12
        coef = 50 * pO2C / self.rl
        std = 0.01 / self.rl
        value[0] = coef * math.exp(-((x[1] - 0.5) ** 2) / (2 * std ** 2))

    def value_shape(self):
        return ()

# -----------------------------------------------------------------------------
# Vertex-wise property assignment (legacy meshes)
# -----------------------------------------------------------------------------


def assign_local_property_vertexBased(
    mesh: Mesh, value: float, V: FunctionSpace
) -> Function:
    kW = Function(V)
    coords = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
    arr = np.full(V.dim(), value)
    # zero out vertices with z > 0
    arr[coords[:, 2] > 0.0] = 0.0
    kW.vector()[:] = arr
    return kW


def assign_local_property_vertexBased_celltags(
    mesh: Mesh,
    value: float,
    V: FunctionSpace,
    cell_tags: MeshFunction,
) -> Function:
    kW = Function(V)
    arr = np.full(V.dim(), value)
    dofmap = V.dofmap()
    for cell in cells(mesh):
        if cell_tags[cell.index()] > 585:
            arr[dofmap.cell_dofs(cell.index())] = 0.0
    kW.vector()[:] = arr
    return kW


def assign_initial_condition_vertex_based(
    mesh: Mesh,
    V: FunctionSpace,
    base_val: float,
    y_thresh: float = 0.3,
) -> Function:
    f = Function(V)
    coords = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
    arr = np.empty(V.dim())
    for i, xyz in enumerate(coords):
        if xyz[2] > 0:
            arr[i] = 0.9 * base_val if xyz[1] < y_thresh else 0.2 * base_val
        else:
            arr[i] = 0.6 * base_val
    f.vector()[:] = arr
    f.vector().apply("insert")
    return f

# -----------------------------------------------------------------------------
# Geometry helpers for direction vectors
# -----------------------------------------------------------------------------


def compute_directional_vectors_cells(mesh: Mesh) -> np.ndarray:
    """Return unit vectors (edge direction) for every cell (1-D pipe mesh)."""
    mesh.init(1, 0)  # ensure edge→vertex conn.
    e2v = mesh.topology()(1, 0)
    coords = mesh.coordinates()
    num_edges = mesh.num_entities(1)
    vecs = np.zeros((num_edges, 3))
    for e in range(num_edges):
        v0, v1 = e2v(e)
        d = coords[v1] - coords[v0]
        n = np.linalg.norm(d)
        vecs[e] = d / n if n > 1e-12 else (0.0, 0.0, 0.0)
    return vecs


class _DirVectorExpr(UserExpression):
    def __init__(self, arr: np.ndarray, **kw):
        super().__init__(**kw)
        self.vecs = arr

    def eval_cell(self, vals, x, cell):  # noqa: D401
        vals[:] = self.vecs[cell.index]

    def value_shape(self):
        return (3,)


def cellDirVec_DG(mesh: Mesh, vecs: np.ndarray) -> Function:
    Vdg = VectorFunctionSpace(mesh, "DG", 0)
    return project(_DirVectorExpr(vecs, degree=0), Vdg)

# -----------------------------------------------------------------------------
# Physiology helpers
# -----------------------------------------------------------------------------


def SHb(domain, CF, solCoef):
    """Hill saturation for haemoglobin (legacy UFL)."""
    P50, n = 26.8, 2.7
    pO2 = CF / Constant(solCoef)
    return (pO2 ** n) / (P50 ** n + pO2 ** n)

# -----------------------------------------------------------------------------
# Minimal PETSc Krylov projection helper (legacy API)
# -----------------------------------------------------------------------------


def project_function_legacy(Vt: FunctionSpace, src) -> Function:
    """Cheap projection using assemble + PETScKrylovSolver (classic dolfin)."""
    u = TrialFunction(Vt)
    v = TestFunction(Vt)
    a = inner(u, v) * dx
    L = inner(src, v) * dx
    A, b = assemble_system(a, L)
    out = Function(Vt)
    solver = PETScKrylovSolver("cg", "ilu")
    solver.set_operator(A)
    solver.solve(out.vector(), b)
    return out


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("auxiliaryFunctions_dolfin (legacy) imported OK")
