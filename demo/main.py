import dolfin as df
import numpy as np
import scipy.io as sio
import os, sys, json
from time import time
from ufl import tanh, variable, max_value
from mpi4py import MPI
from petsc4py import PETSc
from pathlib import Path

base_dir = "/workspace"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# -----------------------------------------------------------------------------
# INITIAL SET-UP
# -----------------------------------------------------------------------------
commMPI = MPI.COMM_WORLD
rank = commMPI.Get_rank()
sizeMPI = commMPI.Get_size()
start_time = time()

# Be conservative with FFC optimizations to avoid segfaults during JIT
try:
    df.parameters["form_compiler"]["cpp_optimize"] = False
    df.parameters["form_compiler"]["optimize"] = False
except Exception:
    pass

# Prefer robust external LU (MUMPS) if available
try:
    df.PETScOptions.set("ksp_type", "preonly")
    df.PETScOptions.set("pc_type", "lu")
    df.PETScOptions.set("pc_factor_mat_solver_type", "mumps")
except Exception:
    pass

# Ensure output folder exists
os.makedirs("./results_1876v", exist_ok=True)
CKPT_META = os.path.join("./results_1876v", "checkpoint_meta.json")
CKPT_H5   = os.path.join("./results_1876v", "checkpoint.h5")

def save_checkpoint(U_mix, step_idx, pseudo_dt_val):
    """Save mixed solution split into components, plus metadata.
    Only rank 0 writes metadata; all ranks participate in HDF5 write.
    """
    comm = MPI.COMM_WORLD
    try:
        Ublood, Utissue = U_mix.split()
        with df.HDF5File(comm, CKPT_H5, "w") as h5:
            Ublood.rename("CFb", "")
            Utissue.rename("CFt", "")
            h5.write(Ublood, "CFb")
            h5.write(Utissue, "CFt")
        if comm.Get_rank() == 0:
            with open(CKPT_META, "w") as f:
                json.dump({
                    "last_completed_step": int(step_idx),  # 0-based
                    "pseudo_dt": float(pseudo_dt_val)
                }, f)
    except Exception as e:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Warning: checkpoint save failed:", repr(e))

def load_checkpoint(U_mix):
    """Load mixed solution from checkpoint if present. Returns (found, last_step, pseudo_dt)."""
    comm = MPI.COMM_WORLD
    if not (os.path.exists(CKPT_META) and os.path.exists(CKPT_H5)):
        return False, -1, None
    try:
        if comm.Get_rank() == 0:
            with open(CKPT_META, "r") as f:
                meta = json.load(f)
        else:
            meta = None
        meta = comm.bcast(meta, root=0)

        CFb_fn = df.Function(V0)
        CFt_fn = df.Function(V1)
        with df.HDF5File(comm, CKPT_H5, "r") as h5:
            h5.read(CFb_fn, "CFb")
            h5.read(CFt_fn, "CFt")
        df.FunctionAssigner(V, [V0, V1]).assign(U_mix, [CFb_fn, CFt_fn])

        return True, int(meta.get("last_completed_step", -1)), float(meta.get("pseudo_dt", 0.1))
    except Exception as e:
        if comm.Get_rank() == 0:
            print("Warning: checkpoint load failed, starting fresh:", repr(e))
        return False, -1, None

# Constants and conversion factors
mL_to_mm = 1000.0                      # mL → mm³
mmHg_to_mmGs = 133.322                 # mmHg → mm g s⁻²
pO2C = 1.35E-12                        # Henry constant

difD_value = 2.41e-5 * 100            # free O₂ diffusivity [mm²/s]
PeCritical = 1                         # SUPG threshold
steadySUPG = 0                        # disable SUPG for stability
Ghypertrophy = 1.0
ratioVtVb = 12.5                       # tissue/ blood volume ratio
kWratioTmp = 0.01                      # wall conductance scaling (further reduced)
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

δ  = df.TrialFunction(V)
phi_b, phi_t = df.TestFunctions(V)     # test functions (blood, tissue)
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
# Project geometric quantities then clamp via vectors (avoid UFL max_value)
Across = df.project(np.pi * Rcell**2,      V_cg)          # cross-section [mm²]
Asurf  = df.project(2.0 * np.pi * Rcell*h, V_cg)          # surface       [mm²]
Vb     = df.project(Across*h,              V_cg)          # blood vol.    [mm³]
Vtis   = Vb * ratioVtVb

# Exchange coefficients (1/s)
kWtmp  = kWratioTmp * 35.0 * 0.001      # [mm/s]
kW     = assign_local_property_vertexBased(mesh, kWtmp, V0)
# Exchange terms with safe division to avoid 0/0
AkVb   = df.project(df.conditional(df.gt(Vb,   df.DOLFIN_EPS), kW*Asurf/Vb,   df.Constant(0.0)), V_cg)
AkVt   = df.project(df.conditional(df.gt(Vtis, df.DOLFIN_EPS), kW*Asurf/Vtis, df.Constant(0.0)), V_cg)

# Advection velocity (scalar)
# Clamp area to epsilon using vector operations
Across_safe = df.Function(V_cg)
_tmp = Across.vector().get_local()
_tmp[_tmp < df.DOLFIN_EPS] = df.DOLFIN_EPS
Across_safe.vector().set_local(_tmp)
Across_safe.vector().apply("insert")

# Compute advection and clamp to epsilon similarly
advU = df.project(Qcell/Across_safe, V_cg)
advU_safe = df.Function(V_cg)
_au = advU.vector().get_local()
_au[_au < df.DOLFIN_EPS] = df.DOLFIN_EPS
# Upper cap to limit extreme velocities (reduce stiffness)
try:
    umax = np.percentile(_au, 95)
    if not np.isfinite(umax) or umax <= df.DOLFIN_EPS:
        umax = 5.0
except Exception:
    umax = 5.0
_au[_au > umax] = umax
advU_safe.vector().set_local(_au)
advU_safe.vector().apply("insert")

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
# Use natural outflow (no Dirichlet at outlet)
bcs    = [bc_in]

# -----------------------------------------------------------------------------
#  SUPG MATRICES
# -----------------------------------------------------------------------------
W      = df.as_matrix([[7/24, -1/24], [13/24, 5/24]])
W_inv  = df.inv(W)
phi_grad = df.dot(df.grad(phi_b), v_dir)
Pw      = W * advU_safe * phi_grad

Pw_vec   = df.as_vector([advU_safe*phi_grad,    # first component
                         advU_safe*phi_grad])   # second component


# -----------------------------------------------------------------------------
#  HELPER: weak form of blood operator
# -----------------------------------------------------------------------------

def weakL(test, CF, CT):
    """Return weak form of blood equation (no SUPG).
    Diffusion applies *only* to dissolved O₂ (CF)."""
    return (
        test * advU_safe * df.dot(df.grad(CT), v_dir)   # advection of total O₂ (safe)
        + Db * df.inner(df.grad(CF), df.grad(test))     # **diffuse dissolved only**
        + test * AkVb * CF                              # exchange source
    )


def funR(CFn, CTn, CFtn):
    ww = df.as_vector([0.5, 0.5])
    # use advU_safe to avoid NaNs in SUPG residual
    return -ww*(AkVb*CFtn - advU_safe*df.dot(df.grad(CTn), v_dir) - AkVb*CFn)

# -----------------------------------------------------------------------------
# 0.  LINEARISED PRE-SOLVE  (assume CT ≈ CF so CB = 0)
# -----------------------------------------------------------------------------
U_lin = df.TrialFunction(V)
CF_lin, CFt_lin = df.split(U_lin)

# blood operator with CT=CF
Fb_lin = (weakL(phi_b, CF_lin, CF_lin) - AkVb * CFt_lin * phi_b) * df.dx

# tissue operator, drop nonlinear uptake term
Ft_lin = (-AkVt * (CF_lin - CFt_lin) * phi_t + Dt * df.inner(df.grad(CFt_lin), df.grad(phi_t))) * df.dx

try:
    a_lin = df.lhs(Fb_lin + Ft_lin)
    L_lin = df.rhs(Fb_lin + Ft_lin)
    # Prefer the most robust pathway first; avoid triggering backend-specific errors
    df.solve(a_lin == L_lin, U_mixed, bcs)
except Exception:
    # Try LinearVariationalSolver, then direct LU, then Krylov; stay quiet unless everything fails
    solved = False
    try:
        problem_lin = df.LinearVariationalProblem(a_lin, L_lin, U_mixed, bcs)
        solver_lin = df.LinearVariationalSolver(problem_lin)
        try:
            solver_lin.parameters["linear_solver"] = "lu"
        except Exception:
            pass
        solver_lin.solve()
        solved = True
    except Exception:
        pass
    if not solved:
        try:
            A_lin, b_lin = df.assemble_system(a_lin, L_lin, bcs)
            lin_solver = df.LUSolver()
            lin_solver.solve(A_lin, U_mixed.vector(), b_lin)
            solved = True
        except Exception:
            pass
    if not solved:
        try:
            A_lin, b_lin = df.assemble_system(a_lin, L_lin, bcs)
            ksp = df.KrylovSolver("gmres", "ilu")
            ksp.parameters["monitor_convergence"] = False
            ksp.parameters["relative_tolerance"] = 1e-10
            ksp.parameters["absolute_tolerance"] = 1e-12
            ksp.parameters["maximum_iterations"] = 500
            ksp.solve(A_lin, U_mixed.vector(), b_lin)
            solved = True
        except Exception as e4:
            if rank == 0:
                print("Linear warm-start failed, continuing with initial guess:", repr(e4))

CF_sol, CFt_sol = U_mixed.split()
F_res = ((weakL(phi_b, CF_sol, CF_sol) - AkVb * CFt_sol * phi_b)
         + (-AkVt * (CF_sol - CFt_sol) * phi_t
            + Dt * df.inner(df.grad(CFt_sol), df.grad(phi_t)))) * df.dx
print("Linear warm-start ‖R‖ =", df.assemble(F_res).norm("l2"))

# -----------------------------------------------------------------------------
#  PSEUDO-TIME LOOP
# -----------------------------------------------------------------------------

maxG_val      = 70e-12 * Ghypertrophy        # [mol mm⁻³ s⁻¹]
num_steps     = 10
pseudo_dt     = df.Constant(0.1)      # stronger transient damping for Newton

# Resume support: if checkpoint exists, load and continue
resume_found, last_step_done, ckpt_dt = load_checkpoint(U_mixed)
start_step = 0
if resume_found:
    start_step = last_step_done + 1
    if ckpt_dt is not None:
        try:
            pseudo_dt.assign(ckpt_dt)
        except Exception:
            pseudo_dt.assign(df.Constant(ckpt_dt))
    # Keep pseudo-time state consistent on resume
    try:
        _, _Ut_resume = U_mixed.split(deepcopy=True)
        Ut_old.assign(_Ut_resume)
    except Exception:
        pass
    if rank == 0:
        print(f"Resuming from checkpoint at step {start_step}/{num_steps}")

_, Ut_old = U_mixed.split(deepcopy=True)

for step in range(start_step, num_steps):
    print(f"\n=== pseudo-time {step+1}/{num_steps} ===")

    # Ramp metabolism: zero on first step, then gradual
    ramp_factor = 0.0 if step == 0 else (step/float(num_steps))
    maxG = assign_local_property_vertexBased(mesh,
                                             maxG_val*ramp_factor,
                                             V0)

    # Ramp wall exchange as well (start tiny), recompute AkVb/AkVt inside loop
    kW_loop = assign_local_property_vertexBased(mesh, kWratioTmp * 35.0 * 0.001 * max(0.05, ramp_factor), V0)
    AkVb   = df.project(df.conditional(df.gt(Vb,   df.DOLFIN_EPS), kW_loop*Asurf/Vb,   df.Constant(0.0)), V_cg)
    AkVt   = df.project(df.conditional(df.gt(Vtis, df.DOLFIN_EPS), kW_loop*Asurf/Vtis, df.Constant(0.0)), V_cg)
    # Cap exchange rates to upper bound to limit stiffness
    Ak_cap = 1e2
    _akb = AkVb.vector().get_local(); _akb = np.clip(_akb, 0.0, Ak_cap)
    AkVb.vector().set_local(_akb); AkVb.vector().apply("insert")
    _akt = AkVt.vector().get_local(); _akt = np.clip(_akt, 0.0, Ak_cap)
    AkVt.vector().set_local(_akt); AkVt.vector().apply("insert")

    # Derived fields (avoid ufl.variable; clamp U to avoid negative fractional powers)
    U_pos = df.conditional(df.ge(U, df.Constant(0.0)), U, df.Constant(0.0))
    CB = 4*CHb*Hct*SHb(mesh, U_pos, pO2C)
    CT = CB + U
    consumption = maxG * Ut / (Ut + km + df.Constant(1e-24))

    # Blood residual (incl. SUPG)
    Fb = (
        weakL(phi_b, U, CT)
        - AkVb * Ut * phi_b
    ) * df.dx

    if steadySUPG:
        # Compute SUPG tau numerically to avoid UFL math on Functions
        _au = advU_safe.vector().get_local()
        _dl = dL.vector().get_local()
        dif = difD_value
        _pe = _au * _dl / (2.0 * (dif + dif/65.0))
        _pe = np.maximum(_pe, 1e-12)
        _sigma = (_dl / (2.0 * _au)) * (1.0/np.tanh(_pe) - 1.0/_pe)
        tau_scalar = df.Function(V_cg)
        tau_scalar.vector().set_local(_sigma)
        tau_scalar.vector().apply("insert")
        tau = tau_scalar * W_inv
        # Fb += df.inner(tau*Pw, funR(U, CT, Ut))*df.dx
        Fb += df.dot(tau*Pw_vec, funR(U, CT, Ut)) * df.dx
    # Tissue residual – **sign fixed** (+AkVt)
    Ft = ((Ut - Ut_old)/pseudo_dt * phi_t
          - AkVt*(U - Ut)*phi_t
          + consumption*phi_t
          + Dt*df.inner(df.grad(Ut), df.grad(phi_t))
          + Dmb*CMb*df.inner(df.grad(Ut/(Ut + C50)), df.grad(phi_t))
          )*df.dx

    F = Fb + Ft
    # Mild artificial diffusion on blood in first step for robustness
    if step == 0:
        F += (0.3*Db) * df.inner(df.grad(U), df.grad(phi_b)) * df.dx
    J = df.derivative(F, U_mixed, δ)

    # Newton solve with DOLFIN's NonlinearVariationalSolver
    try:
        Fb_vec = df.assemble(Fb)
        Ft_vec = df.assemble(Ft)
        if rank == 0:
            print("||Fb|| =", Fb_vec.norm("l2"), " ||Ft|| =", Ft_vec.norm("l2"))
    except Exception as e:
        if rank == 0:
            print("Assembly check failed:", repr(e))

    problem = df.NonlinearVariationalProblem(F, U_mixed, bcs, J)
    solver = df.NonlinearVariationalSolver(problem)
    prm = solver.parameters
    try:
        prm["newton_solver"]["relative_tolerance"] = 1e-6
        prm["newton_solver"]["absolute_tolerance"] = 1e-8
        prm["newton_solver"]["maximum_iterations"] = 50
        prm["newton_solver"]["linear_solver"] = "lu"
        prm["newton_solver"]["relaxation_parameter"] = 0.3
        prm["newton_solver"]["error_on_nonconvergence"] = False
        prm["newton_solver"]["line_search"] = "bt"
    except Exception:
        pass

    # Retry with pseudo_dt backoff if failure occurs
    max_retries = 5
    try_id = 0
    success_local = 0
    current_dt = None
    try:
        current_dt = float(pseudo_dt.values()[0])
    except Exception:
        # Fallback if Constant API differs
        current_dt = 0.1
    U_prev = U_mixed.copy(deepcopy=True)
    while try_id < max_retries:
        # Update Constant in case it changed
        try:
            pseudo_dt.assign(current_dt)
        except Exception:
            pseudo_dt.assign(df.Constant(current_dt))

        try:
            solver.solve()
            vec = U_mixed.vector().get_local()
            if np.all(np.isfinite(vec)):
                success_local = 1
            else:
                success_local = 0
        except Exception as e:
            success_local = 0

        # MPI agreement: all ranks must succeed
        success_global = MPI.COMM_WORLD.allreduce(success_local, op=MPI.MIN)
        if success_global == 1:
            break

        # Revert and back off dt, clamp negatives
        df.FunctionAssigner(V, [V0, V1]).assign(U_mixed, U_prev.split())
        vloc = U_mixed.vector().get_local()
        vloc[~np.isfinite(vloc)] = 0.0
        vloc[vloc < 0.0] = 0.0
        U_mixed.vector().set_local(vloc)
        U_mixed.vector().apply("insert")

        current_dt = max(current_dt/2.0, 1e-4)
        if rank == 0:
            print(f"Solve failed on try {try_id+1}; reducing dt to {current_dt:.4e} and retrying.")
        try_id += 1

    if try_id == max_retries and success_local == 0:
        if rank == 0:
            print("Step failed after retries; stopping early. Last completed step is saved.")
        # Save the last successful checkpoint of previous step is already on disk
        break

    # Update pseudo-time variable
    _, Ut_new = U_mixed.split(deepcopy=True)
    Ut_old.assign(Ut_new)

    # Write results and checkpoint
    sid = step+1
    with df.XDMFFile(commMPI, f"./results_1876v/CFb_step_{sid:02d}.xdmf") as xb:
        Ublood, _ = U_mixed.split()
        Ublood.rename("CFb", "")
        xb.write(Ublood)
    with df.XDMFFile(commMPI, f"./results_1876v/CFt_step_{sid:02d}.xdmf") as xt:
        _, Utissue = U_mixed.split()
        Utissue.rename("CFt", "")
        xt.write(Utissue)

    # Persist checkpoint for resume (0-based step index)
    save_checkpoint(U_mixed, step, current_dt)

print(f"Total runtime: {time() - start_time:.1f} s")
