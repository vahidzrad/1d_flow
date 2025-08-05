import dolfin as df
import numpy as np
import scipy.io as sio
import os, sys
from time import time
from ufl import tanh, dot, grad, inner, variable
from mpi4py import MPI
from petsc4py import PETSc


# base_dir = "/mnt/home/ziaeirad/1d_flow/"
base_dir = "/workspace"
# base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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
mesh = df.Mesh()
with df.XDMFFile(commMPI, os.path.join(base_dir, "mesh", "1876v_90TV_dL0.001_2tags.xdmf")) as infile:
    infile.read(mesh)
    mvc_cells = df.MeshValueCollection("size_t", mesh, mesh.topology().dim())
    infile.read(mvc_cells, "Cell tags")
    cell_tags = df.cpp.mesh.MeshFunctionSizet(mesh, mvc_cells)
    mvc_vertices = df.MeshValueCollection("size_t", mesh, 0)
    infile.read(mvc_vertices, "mesh_tags")
    vertex_tags = df.MeshFunction("size_t", mesh, mvc_vertices)

INLET_TAG  = 1
OUTLET_TAG = 2  # add more outlet tags here if needed

# -----------------------------------------------------------------------------
# FUNCTION SPACES
# -----------------------------------------------------------------------------
P1 = df.FiniteElement("CG", mesh.ufl_cell(), 1)
V  = df.FunctionSpace(mesh, df.MixedElement([P1, P1]))
V0, V1 = V.sub(0).collapse(), V.sub(1).collapse()


# Mixed unknown function
U_mixed = df.Function(V)

ψ  = df.TestFunction(V)
δ  = df.TrialFunction(V)
ϕ, ϕt = df.split(ψ)            # test functions (blood, tissue)
# U,  Ut = U_mixed.split()    # unknowns  (blood, tissue)
U, Ut = df.split(U_mixed)

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
V_dg  = df.FunctionSpace(mesh, "DG", 0)
Qcell = df.Function(V_dg)
Rcell = df.Function(V_dg)

cell_ids  = np.array([c.index() for c in df.cells(mesh)], dtype=int)
cell_vids = np.array([cell_tags[c] for c in df.cells(mesh)], dtype=int) - 1
Qcell.vector().set_local(Qvessel[cell_vids])
Rcell.vector().set_local(Rvessel[cell_vids])
Qcell.vector().apply("insert")
Rcell.vector().apply("insert")

# CG1 projections → vertex values
V_cg   = df.FunctionSpace(mesh, "CG", 1)
Qnode  = df.project(Qcell,  V_cg)
Rnode  = df.project(Rcell,  V_cg)

# Geometry helpers
h      = df.CellDiameter(mesh)
dL     = df.project(h, V_cg)
Across = df.project(np.pi * Rcell**2,      V_cg)          # cross-section [mm²]
Asurf  = df.project(2.0 * np.pi * Rcell*h, V_cg)          # surface       [mm²]
Vb     = df.project(Across*h,              V_cg)          # blood vol.    [mm³]
Vtis   = Vb * ratioVtVb

# Exchange coefficients (1/s)
kWtmp  = kWratioTmp * 35.0 * 0.001      # [mm/s]
kW     = assign_local_property_vertexBased(mesh, kWtmp, V0)
AkVb   = df.project(kW*Asurf/ Vb,           V_cg)
AkVt   = df.project(kW*Asurf/ Vtis,         V_cg)

# Advection velocity (scalar)
advU   = df.project(Qcell/Across, V_cg)
# Floor advection to avoid singular SUPG tau when flow is locally zero
advU_safe = df.conditional(df.gt(advU, df.DOLFIN_EPS), advU, df.DOLFIN_EPS)

# Vessel direction vectors CG1
v_dir_DG = cellDirVec_DG(mesh, compute_directional_vectors_cells(mesh))
v_dir    = df.project(v_dir_DG, df.VectorFunctionSpace(mesh, "CG", 1))
if sizeMPI > 1:
    v_dir.vector.ghostUpdate(addv=df.PETSc.InsertMode.INSERT,
                             mode=df.PETSc.ScatterMode.FORWARD)
    commMPI.barrier()

# -----------------------------------------------------------------------------
#  CONSTANTS
# -----------------------------------------------------------------------------
Db  = df.Constant(difD_value)
Dt  = df.Constant(difD_value)
Dmb = df.Constant(2.2e-7 * 100)
CHb = df.Constant(5.3e-9)
Hct = df.Constant(HctTmp)
km  = df.Constant(1e-7 * 1e-6)
CMb = df.Constant(1e-4 * 1e-6)
C50 = df.Constant(2.5 * mmHg_to_mmGs)

# -----------------------------------------------------------------------------
#  INITIAL CONDITIONS
# -----------------------------------------------------------------------------
U_init  = assign_initial_condition_vertex_based(mesh, V0, 100*pO2C)
Ut_init = df.interpolate(df.Constant(50*pO2C), V1)

df.FunctionAssigner(V, [V0, V1]).assign(U_mixed, [U_init, Ut_init])

# -----------------------------------------------------------------------------
#  BOUNDARY CONDITIONS
# -----------------------------------------------------------------------------
bc_in  = df.DirichletBC(V.sub(0), df.Constant(100*pO2C), vertex_tags, INLET_TAG)
bc_out = df.DirichletBC(V.sub(0), df.Constant( 20*pO2C), vertex_tags, OUTLET_TAG)
bcs    = [bc_in, bc_out]

# -----------------------------------------------------------------------------
#  SUPG MATRICES
# -----------------------------------------------------------------------------
W      = df.as_matrix([[7/24, -1/24], [13/24, 5/24]])
W_inv  = df.inv(W)
phi_grad = dot(grad(ϕ), v_dir)
Pw      = W * advU * phi_grad

Pw_vec   = df.as_vector([advU*phi_grad,    # first component
                         advU*phi_grad])   # second component


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
    ww = df.as_vector([0.5, 0.5])
    return -ww*(AkVb*CFtn - advU*dot(grad(CTn), v_dir) - AkVb*CFn)

# -----------------------------------------------------------------------------
# 0.  LINEARISED PRE-SOLVE  (assume CT ≈ CF so CB = 0)
# -----------------------------------------------------------------------------
U_lin = df.TrialFunction(V)
CF_lin, CFt_lin = df.split(U_lin)

# blood operator with CT=CF
Fb_lin = (weakL(ϕ, CF_lin, CF_lin) - AkVb * CFt_lin * ϕ) * df.dx

# tissue operator, drop nonlinear uptake term
Ft_lin = (-AkVt * (CF_lin - CFt_lin) * ϕt + Dt * inner(grad(CFt_lin), grad(ϕt))) * df.dx

a_lin = df.lhs(Fb_lin + Ft_lin)
L_lin = df.rhs(Fb_lin + Ft_lin)
df.solve(a_lin == L_lin, U_mixed, bcs)

CF_sol, CFt_sol = U_mixed.split()
F_res = ((weakL(ϕ, CF_sol, CF_sol) - AkVb * CFt_sol * ϕ)
         + (-AkVt * (CF_sol - CFt_sol) * ϕt
            + Dt * inner(grad(CFt_sol), grad(ϕt)))) * df.dx
print("Linear warm-start ‖R‖ =", df.assemble(F_res).norm("l2"))

# -----------------------------------------------------------------------------
#  PSEUDO-TIME LOOP
# -----------------------------------------------------------------------------

maxG_val      = 70e-12 * Ghypertrophy        # [mol mm⁻³ s⁻¹]
num_steps     = 10
pseudo_dt     = df.Constant(1e2)

_, Ut_old = U_mixed.split(deepcopy=True)

for step in range(num_steps):
    print(f"\n=== pseudo-time {step+1}/{num_steps} ===")

    # Ramp metabolism
    maxG = assign_local_property_vertexBased(mesh,
                                             maxG_val*(step+1)/num_steps,
                                             V0)

    # Derived fields
    CB = variable(4*CHb*Hct*SHb(mesh, U, pO2C))
    CT = CB + U
    consumption = variable(maxG * Ut / (Ut + km + 1e-24))

    # Blood residual (incl. SUPG)
    Fb = (
        weakL(ϕ, U, CT)
        - AkVb * Ut * ϕ
    ) * df.dx

    if steadySUPG:
        Pe = advU_safe*dL / (2*(Db + Db/65))
        tau = (dL/(2*advU_safe))*(1/tanh(Pe) - 1/Pe) * W_inv
        # Fb += inner(tau*Pw, funR(U, CT, Ut))*df.dx   # inner-product fix
        Fb += dot(tau*Pw_vec, funR(U, CT, Ut)) * df.dx   # or inner(tauPw, funR(...))*df.dx
    # Tissue residual – **sign fixed** (+AkVt)
    Ft = ((Ut - Ut_old)/pseudo_dt * ϕt
          + AkVt*(U - Ut)*ϕt                     # ← sign corrected
          + consumption*ϕt
          + Dt*inner(grad(Ut), grad(ϕt))
          + Dmb*CMb*inner(grad(Ut/(Ut + C50)), grad(ϕt))
          )*df.dx

    F = Fb + Ft
    J = df.derivative(F, U_mixed, δ)

    # -----------------------------------------------
    # TAO nonlinear solve
    # -----------------------------------------------
    A = df.PETScMatrix(); df.assemble(J, tensor=A)
    b = df.PETScVector(); df.assemble(F, tensor=b)
    x = df.as_backend_type(U_mixed.vector()).vec()
    tao = PETSc.TAO().create(commMPI)
    tao.setType("tron")

    def objective(tao_, x_):
        x_.copy(result=x)
        df.assemble(F, tensor=b)
        return 0.5 * b.norm("l2")**2

    def gradient(tao_, x_, g_):
        x_.copy(result=x)
        df.assemble(F, tensor=b)
        df.assemble(J, tensor=A)
        Amat = df.as_backend_type(A).mat()
        Amat.multTranspose(b.vec(), g_)

    def hessian(tao_, x_, H_, P_):
        x_.copy(result=x)
        df.assemble(J, tensor=A)
        Amat = df.as_backend_type(A).mat()
        JtJ = Amat.transposeMatMult(Amat)
        H_.setValuesCSR(*JtJ.getValuesCSR())
        H_.assemble()

    tao.setObjective(objective)
    tao.setGradient(gradient)
    tao.setHessian(hessian)

    ksp = tao.getKSP()
    ksp.setType("preonly")
    ksp.getPC().setType("lu")

    tao.setTolerances(gatol=1e-8, grtol=1e-6)
    tao.setFromOptions()
    tao.solve(x)

    # Update pseudo-time variable
    _, Ut_new = U_mixed.split(deepcopy=True)
    Ut_old.assign(Ut_new)

    # Write results
    sid = step+1
    with df.XDMFFile(commMPI, f"./results_1876v/CFb_step_{sid:02d}.xdmf") as xb:
        Ublood, _ = U_mixed.split()
        Ublood.rename("CFb", "")
        xb.write(Ublood)
    with df.XDMFFile(commMPI, f"./results_1876v/CFt_step_{sid:02d}.xdmf") as xt:
        _, Utissue = U_mixed.split()
        Utissue.rename("CFt", "")
        xt.write(Utissue)




























print(f"Total runtime: {time() - start_time:.1f} s")
