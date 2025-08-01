"""Simplified pseudo-time stepping solver for the 1D flow model.

This script mirrors the original implementation but wraps the most
important set-up steps in small helper functions.  The aim is easier
maintenance and clearer separation of mesh loading, data preparation
and the actual solver loop.

The hard coded ``base_dir`` from the original version is preserved as
requested.  Only the surrounding logic is refactored.
"""

from dolfin import *
import numpy as np
import scipy.io as sio
import os
import sys
from time import time
from ufl import tanh, dot, grad, inner, variable
from mpi4py import MPI
import argparse

base_dir = "/mnt/home/ziaeirad/1d_flow/"
sys.path.append(base_dir)
sys.path.append(base_dir + "src")

# -----------------------------------------------------------------------------
# INITIAL SET-UP
# -----------------------------------------------------------------------------
commMPI = MPI.COMM_WORLD
rank = commMPI.Get_rank()
sizeMPI = commMPI.Get_size()
start_time = time()

# Constants and conversion factors
mL_to_mm = 1000.0                      # mL → mm³
mmHg_to_mmGs = 133.322                 # mmHg → mm g s⁻²
pO2C = 1.35E-12                        # Henry constant

difD_value = 2.41e-5 * 100            # free O₂ diffusivity [mm²/s]
PeCritical = 1                         # SUPG threshold
steadySUPG = 1
Ghypertrophy = 1.0
ratioVtVb = 12.5                       # tissue/ blood volume ratio
kWratioTmp = 1.0                       # wall conductance scaling
HctTmp = 0.25                          # haematocrit

# -----------------------------------------------------------------------------
# MESH & TAGS
# -----------------------------------------------------------------------------
mesh = Mesh()
with XDMFFile(commMPI, os.path.join(base_dir, "mesh", "1876v_90TV_dL0.001_2tags.xdmf")) as infile:
    infile.read(mesh)
    mvc_cells = MeshValueCollection("size_t", mesh, mesh.topology().dim())
    infile.read(mvc_cells, "Cell tags")
    cell_tags = cpp.mesh.MeshFunctionSizet(mesh, mvc_cells)
    mvc_vertices = MeshValueCollection("size_t", mesh, 0)
    infile.read(mvc_vertices, "mesh_tags")
    vertex_tags = MeshFunction("size_t", mesh, mvc_vertices)

INLET_TAG  = 1
OUTLET_TAG = 2  # add more outlet tags here if needed

# -----------------------------------------------------------------------------
# FUNCTION SPACES
# -----------------------------------------------------------------------------
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
V  = FunctionSpace(mesh, MixedElement([P1, P1]))
V0, V1 = V.sub(0).collapse(), V.sub(1).collapse()

ψ  = TestFunction(V)
δ  = TrialFunction(V)
ϕ, ϕt = split(ψ)            # test functions (blood, tissue)
U,  Ut = split(Function(V))  # unknowns  (blood, tissue)

# Ensure non-negative arguments in nonlinear terms
U_safe  = conditional(gt(U,  0.0), U,  0.0)
Ut_safe = conditional(gt(Ut, 0.0), Ut, 0.0)

# -----------------------------------------------------------------------------
#  AUXILIARY DATA (helpers from external module)
# -----------------------------------------------------------------------------
from auxiliaryFunctions_dolfin import (
    compute_directional_vectors_cells,
    cellDirVec_DG,
    SHb,
    assign_local_property_vertexBased,
    assign_initial_condition_vertex_based,
)

# -----------------------------------------------------------------------------
#  HEMODYNAMIC DATA (MAT-files)
# -----------------------------------------------------------------------------
mat = sio.loadmat(os.path.join(base_dir, "matlabData", "1786v_90TV", "DataSaved.mat"))
Qvessel = mat["Qio"].astype(float)[-1] * 1000.0   # flow [mm³/s]
Rvessel = mat["Rat"].astype(float)[-1]

# Discontinuous cell-wise spaces for Q & R
V_dg  = FunctionSpace(mesh, "DG", 0)
Qcell = Function(V_dg)
Rcell = Function(V_dg)

cell_ids  = np.array([c.index() for c in cells(mesh)], dtype=int)
cell_vids = np.array([cell_tags[c] for c in cells(mesh)], dtype=int) - 1
Qcell.vector().set_local(Qvessel[cell_vids])
Rcell.vector().set_local(Rvessel[cell_vids])
Qcell.vector().apply("insert")
Rcell.vector().apply("insert")

# CG1 projections → vertex values
V_cg   = FunctionSpace(mesh, "CG", 1)
Qnode  = project(Qcell,  V_cg)
Rnode  = project(Rcell,  V_cg)

# Geometry helpers
h      = CellDiameter(mesh)
dL     = project(h, V_cg)
Across = project(np.pi * Rcell**2,      V_cg)          # cross-section [mm²]
Asurf  = project(2.0 * np.pi * Rcell*h, V_cg)          # surface       [mm²]
Vb     = project(Across*h,              V_cg)          # blood vol.    [mm³]
Vtis   = Vb * ratioVtVb

# Exchange coefficients (1/s)
kWtmp  = kWratioTmp * 35.0 * 0.001      # [mm/s]
kW     = assign_local_property_vertexBased(mesh, kWtmp, V0)
AkVb   = project(kW*Asurf/ Vb,           V_cg)
AkVt   = project(kW*Asurf/ Vtis,         V_cg)

# Advection velocity (scalar)
advU   = project(Qcell/Across, V_cg)

# Vessel direction vectors CG1
v_dir_DG = cellDirVec_DG(mesh, compute_directional_vectors_cells(mesh))
v_dir    = project(v_dir_DG, VectorFunctionSpace(mesh, "CG", 1))
if sizeMPI > 1:
    v_dir.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                             mode=PETSc.ScatterMode.FORWARD)
    commMPI.barrier()

# -----------------------------------------------------------------------------
#  CONSTANTS
# -----------------------------------------------------------------------------
Db  = Constant(difD_value)
Dt  = Constant(difD_value)
Dmb = Constant(2.2e-7 * 100)
CHb = Constant(5.3e-9)
Hct = Constant(HctTmp)
km  = Constant(1e-7 * 1e-6)
CMb = Constant(1e-4 * 1e-6)
C50 = Constant(2.5 * mmHg_to_mmGs)

# -----------------------------------------------------------------------------
#  INITIAL CONDITIONS
# -----------------------------------------------------------------------------
U_init  = assign_initial_condition_vertex_based(mesh, V0, 100*pO2C)
Ut_init = interpolate(Constant(50*pO2C), V1)
FunctionAssigner(V, [V0, V1]).assign(U, [U_init, Ut_init])

# -----------------------------------------------------------------------------
#  BOUNDARY CONDITIONS
# -----------------------------------------------------------------------------
bc_in  = DirichletBC(V.sub(0), Constant(100*pO2C), vertex_tags, INLET_TAG)
bc_out = DirichletBC(V.sub(0), Constant( 20*pO2C), vertex_tags, OUTLET_TAG)
bcs    = [bc_in, bc_out]

# -----------------------------------------------------------------------------
#  SUPG MATRICES
# -----------------------------------------------------------------------------
W      = as_matrix([[7/24, -1/24], [13/24, 5/24]])
W_inv  = inv(W)
phi_grad = dot(grad(ϕ), v_dir)
Pw      = W * advU * phi_grad

# -----------------------------------------------------------------------------
#  HELPER: weak form of blood operator
# -----------------------------------------------------------------------------

def weakL(test, CF, CT):
    """Return weak form of blood equation (no SUPG).
    Diffusion applies *only* to dissolved O₂ (CF)."""
    return (
        test * advU * dot(grad(CT), v_dir)        # advection of total O₂
        + Db * inner(grad(CF), grad(test))        # **diffuse dissolved only**
        + test * AkVb * CF                        # exchange source
    )


def funR(CFn, CTn, CFtn):
    ww = as_vector([0.5, 0.5])
    return -ww*(AkVb*CFtn - advU*dot(grad(CTn), v_dir) - AkVb*CFn)

# -----------------------------------------------------------------------------
#  PSEUDO-TIME LOOP
# -----------------------------------------------------------------------------

def pseudo_time_loop(num_steps=10, pseudo_dt=Constant(1e2)):
    """Run the non-linear pseudo-time iterations."""
    start_time = time()
    _, Ut_old = U.split(deepcopy=True)

    for step in range(num_steps):
        print(f"\n=== pseudo-time {step+1}/{num_steps} ===")

        # Ramp metabolism
        maxG = assign_local_property_vertexBased(
            mesh, maxG_val * (step + 1) / num_steps, V0
        )

        # Derived fields
        CB = variable(4 * CHb * Hct * SHb(mesh, U_safe, pO2C))
        CT = CB + U_safe
        consumption = maxG * Ut_safe / (Ut_safe + km + 1e-24)

        # Blood residual (incl. SUPG)
        Fb = (
            0.5 * weakL(ϕ, U_safe, CT)
            + 0.5 * weakL(ϕ, U_safe, CT)
            - (0.5 * AkVb * Ut * ϕ + 0.5 * AkVb * Ut * ϕ) * dx
        )

        if steadySUPG:
            Pe = advU * dL / (2 * (Db + Db / 65))
            tau = (dL / (2 * advU)) * (1 / tanh(Pe) - 1 / Pe) * W_inv
            Fb += inner(tau * Pw, funR(U, CT, Ut)) * dx

        # Tissue residual – **sign fixed** (+AkVt)
        Ft = (
            (Ut - Ut_old) / pseudo_dt * ϕt
            + AkVt * (U - Ut) * ϕt
            + consumption * ϕt
            + Dt * inner(grad(Ut), grad(ϕt))
            + Dmb * CMb * inner(grad(Ut / (Ut + C50)), grad(ϕt))
        ) * dx

        F = Fb + Ft
        J = derivative(F, U, δ)

        problem = NonlinearVariationalProblem(F, U, bcs, J)
        solver = NonlinearVariationalSolver(problem)
        prm = solver.parameters["snes_solver"]
        prm.update(
            {
                "report": True,
                "absolute_tolerance": 1e-13,
                "maximum_iterations": 1000,
                "linear_solver": "lu",
            }
        )

        solver.solve()

        # Update pseudo-time variable
        _, Ut_new = U.split(deepcopy=True)
        Ut_old.assign(Ut_new)

        # Write results
        sid = step + 1
        with XDMFFile(commMPI, f"./results_1876v/CFb_step_{sid:02d}.xdmf") as xb:
            Ublood, _ = U.split()
            Ublood.rename("CFb", "")
            xb.write(Ublood)
        with XDMFFile(commMPI, f"./results_1876v/CFt_step_{sid:02d}.xdmf") as xt:
            _, Utissue = U.split()
            Utissue.rename("CFt", "")
            xt.write(Utissue)

    print(f"Total runtime: {time() - start_time:.1f} s")


maxG_val = 70e-12 * Ghypertrophy  # [mol mm⁻³ s⁻¹]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pseudo-time solver")
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="number of pseudo-time steps"
    )
    args = parser.parse_args()
    pseudo_time_loop(num_steps=args.steps)
