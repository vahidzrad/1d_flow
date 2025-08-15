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

# -----------------------------------------------------------------------------
# DIAGNOSTICS: print MPI/PETSc vendor and selected PETSc options per rank
# -----------------------------------------------------------------------------
def print_diagnostics(prefix="Startup"):
    try:
        mpi_ver = MPI.Get_library_version().strip()
    except Exception:
        mpi_ver = "<unknown>"
    try:
        petsc_ver = PETSc.Sys.getVersion()
    except Exception:
        petsc_ver = "<unknown>"
    try:
        petsc_vendor = PETSc.Sys.getVendor()
    except Exception:
        petsc_vendor = ("<unknown>", "", "")
    try:
        opts = PETSc.Options()
        ksp_type = opts.getString("ksp_type")
        pc_type = opts.getString("pc_type")
        pc_factor = opts.getString("pc_factor_mat_solver_type")
    except Exception:
        ksp_type = pc_type = pc_factor = None

    msg = (
        f"R{rank}/{sizeMPI} [{prefix}] "
        f"MPI='{mpi_ver}', PETSc='{petsc_ver}', Vendor={petsc_vendor}; "
        f"ksp_type={ksp_type}, pc_type={pc_type}, pc_factor={pc_factor}"
    )
    print(msg)
    sys.stdout.flush()

print_diagnostics("Startup")

# Rank 0: show argv to confirm PETSc-style flags are reaching Python
if rank == 0:
    try:
        print(f"argv: {sys.argv}")
        sys.stdout.flush()
    except Exception:
        pass

# Allow simple overrides via argv/env (use alongside PETSc flags if needed)
force_ksp = None
force_pc = None
skip_hypre = os.environ.get("FENICS_SKIP_HYPRE", "0") == "1" or "--skip-hypre" in sys.argv
for i, a in enumerate(list(sys.argv)):
    if a == "--ksp" and i + 1 < len(sys.argv):
        force_ksp = sys.argv[i + 1]
    if a == "--pc" and i + 1 < len(sys.argv):
        force_pc = sys.argv[i + 1]

# Global defaults for implicit solves (df.project, etc.) in MPI runs
if sizeMPI > 1:
    try:
        df.parameters["linear_solver"] = "gmres"
        df.parameters["preconditioner"] = "ilu"
    except Exception:
        pass


# Be conservative with FFC optimizations to avoid segfaults during JIT
try:
    df.parameters["form_compiler"]["cpp_optimize"] = False
    df.parameters["form_compiler"]["optimize"] = False
except Exception:
    pass

# Linear solver defaults: avoid forcing LU/MUMPS under MPI to prevent hangs
try:
    if MPI.COMM_WORLD.Get_size() == 1:
        df.PETScOptions.set("ksp_type", "preonly")
        df.PETScOptions.set("pc_type", "lu")
        df.PETScOptions.set("pc_factor_mat_solver_type", "mumps")
    else:
        # Reasonable parallel-safe defaults; prefer GAMG to avoid HYPRE dependency
        df.PETScOptions.set("ksp_type", "gmres")
        df.PETScOptions.set("pc_type", "gamg")
        df.PETScOptions.set("ksp_rtol", 1e-8)
        df.PETScOptions.set("ksp_atol", 1e-12)
        df.PETScOptions.set("ksp_max_it", 500)
    # Apply user overrides if provided
    if force_ksp:
        df.PETScOptions.set("ksp_type", force_ksp)
    if force_pc:
        df.PETScOptions.set("pc_type", force_pc)
except Exception:
    pass

# Print options after we possibly set defaults/overrides
print_diagnostics("AfterOpts")

# Ensure output folder exists
os.makedirs("./results_1876v", exist_ok=True)
CKPT_META = os.path.join("./results_1876v", "checkpoint_meta.json")
CKPT_H5 = os.path.join("./results_1876v", "checkpoint.h5")


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
                json.dump(
                    {
                        "last_completed_step": int(step_idx),  # 0-based
                        "pseudo_dt": float(pseudo_dt_val),
                    },
                    f,
                )
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

        return (
            True,
            int(meta.get("last_completed_step", -1)),
            float(meta.get("pseudo_dt", 0.1)),
        )
    except Exception as e:
        if comm.Get_rank() == 0:
            print("Warning: checkpoint load failed, starting fresh:", repr(e))
        return False, -1, None


# Constants and conversion factors
mL_to_mm = 1000.0  # mL → mm³
mmHg_to_mmGs = 133.322  # mmHg → mm g s⁻²
pO2C = 1.35e-12  # Henry constant

difD_value = 2.41e-5 * 100  # free O₂ diffusivity [mm²/s]
PeCritical = 1  # SUPG threshold
steadySUPG = 0  # disable SUPG for stability
Ghypertrophy = 1.0
ratioVtVb = 12.5  # tissue/ blood volume ratio
kWratioTmp = 0.1  # wall conductance scaling (match main_param)
HctTmp = 0.25  # haematocrit

# -----------------------------------------------------------------------------
# MESH & TAGS
# -----------------------------------------------------------------------------
mesh = df.Mesh()
with df.XDMFFile(
    commMPI, os.path.join(base_dir, "mesh", "1876v_90TV_dL0.001_2tags.xdmf")
) as infile:
    infile.read(mesh)
    mvc_cells = df.MeshValueCollection("size_t", mesh, mesh.topology().dim())
    infile.read(mvc_cells, "Cell tags")
    cell_tags = df.cpp.mesh.MeshFunctionSizet(mesh, mvc_cells)
    mvc_vertices = df.MeshValueCollection("size_t", mesh, 0)
    infile.read(mvc_vertices, "mesh_tags")
    vertex_tags = df.MeshFunction("size_t", mesh, mvc_vertices)

INLET_TAG = 1
OUTLET_TAG = 2  # add more outlet tags here if needed

# -----------------------------------------------------------------------------
# FUNCTION SPACES
# -----------------------------------------------------------------------------
P1 = df.FiniteElement("CG", mesh.ufl_cell(), 1)
V = df.FunctionSpace(mesh, df.MixedElement([P1, P1]))
V0, V1 = V.sub(0).collapse(), V.sub(1).collapse()


# Mixed unknown function
U_mixed = df.Function(V)

δ = df.TrialFunction(V)
phi_b, phi_t = df.TestFunctions(V)  # test functions (blood, tissue)
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
Qvessel = mat["Qio"].astype(float)[-1] * 1000.0  # flow [mm³/s]
Rvessel = mat["Rat"].astype(float)[-1]

# Discontinuous cell-wise spaces for Q & R
V_dg = df.FunctionSpace(mesh, "DG", 0)
Qcell = df.Function(V_dg)
Rcell = df.Function(V_dg)

cell_ids = np.array([c.index() for c in df.cells(mesh)], dtype=int)
cell_vids = np.array([cell_tags[c] for c in df.cells(mesh)], dtype=int) - 1
Qcell.vector().set_local(Qvessel[cell_vids])
Rcell.vector().set_local(Rvessel[cell_vids])
Qcell.vector().apply("insert")
Rcell.vector().apply("insert")

# CG1 projections → vertex values
V_cg = df.FunctionSpace(mesh, "CG", 1)
Qnode = df.project(Qcell, V_cg)
Rnode = df.project(Rcell, V_cg)

# Geometry helpers
h = df.CellDiameter(mesh)
dL = df.project(h, V_cg)
# Project geometric quantities then clamp via vectors (avoid UFL max_value)
Across = df.project(np.pi * Rcell**2, V_cg)  # cross-section [mm²]
Asurf = df.project(2.0 * np.pi * Rcell * h, V_cg)  # surface       [mm²]
Vb = df.project(Across * h, V_cg)  # blood vol.    [mm³]
Vtis = Vb * ratioVtVb

# Exchange coefficients (1/s)
kWtmp = kWratioTmp * 35.0 * 0.001  # [mm/s]
kW = assign_local_property_vertexBased(mesh, kWtmp, V0)
# Exchange terms with safe division to avoid 0/0
AkVb = df.project(
    df.conditional(df.gt(Vb, df.DOLFIN_EPS), kW * Asurf / Vb, df.Constant(0.0)), V_cg
)
AkVt = df.project(
    df.conditional(df.gt(Vtis, df.DOLFIN_EPS), kW * Asurf / Vtis, df.Constant(0.0)),
    V_cg,
)

# Advection velocity (scalar)
# Clamp area to epsilon using vector operations
Across_safe = df.Function(V_cg)
_tmp = Across.vector().get_local()
_tmp[_tmp < df.DOLFIN_EPS] = df.DOLFIN_EPS
Across_safe.vector().set_local(_tmp)
Across_safe.vector().apply("insert")

# Compute advection and clamp to epsilon similarly
advU = df.project(Qcell / Across_safe, V_cg)
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
v_dir = df.project(v_dir_DG, df.VectorFunctionSpace(mesh, "CG", 1))
if sizeMPI > 1:
    # In dolfin, finalize assembly and update ghosts via apply("insert")
    v_dir.vector().apply("insert")
    commMPI.barrier()

# -----------------------------------------------------------------------------
#  CONSTANTS
# -----------------------------------------------------------------------------
Db = df.Constant(difD_value)
Dt = df.Constant(difD_value)
Dmb = df.Constant(2.2e-7 * 100)
CHb = df.Constant(5.3e-9 / pO2C)
Hct = df.Constant(HctTmp)
km = df.Constant(0.1 / 1.35)
CMb = df.Constant(1e-4 * 1e-6 / (pO2C))
C50 = df.Constant(2.5)

# -----------------------------------------------------------------------------
#  INITIAL CONDITIONS
# -----------------------------------------------------------------------------
U_init = assign_initial_condition_vertex_based(mesh, V0, 100)
Ut_init = df.interpolate(df.Constant(50), V1)

df.FunctionAssigner(V, [V0, V1]).assign(U_mixed, [U_init, Ut_init])

# -----------------------------------------------------------------------------
#  BOUNDARY CONDITIONS
# -----------------------------------------------------------------------------
no_bc = ("--no-bc" in sys.argv)
if no_bc and rank == 0:
    print("BCs disabled via --no-bc (debug mode)")
if not no_bc:
    bc_in = df.DirichletBC(V.sub(0), df.Constant(100), vertex_tags, INLET_TAG)
    bc_out = df.DirichletBC(V.sub(0), df.Constant(20), vertex_tags, OUTLET_TAG)
    bcs = [bc_in, bc_out]
else:
    bcs = []

# -----------------------------------------------------------------------------
#  SUPG MATRICES
# -----------------------------------------------------------------------------
W = df.as_matrix([[7 / 24, -1 / 24], [13 / 24, 5 / 24]])
W_inv = df.inv(W)
phi_grad = df.dot(df.grad(phi_b), v_dir)
# Use raw advU for SUPG weighting (match main_param behavior)
Pw = W * advU * phi_grad

Pw_vec = df.as_vector(
    [advU * phi_grad, advU * phi_grad]  # first component
)  # second component


# -----------------------------------------------------------------------------
#  HELPER: weak form of blood operator
# -----------------------------------------------------------------------------


def weakL(test, CF, CT):
    """Return weak form of blood equation (no SUPG).
    Diffusion applies *only* to dissolved O₂ (CF)."""
    return (
        test * advU * df.dot(df.grad(CT), v_dir)  # advection of total O₂ (raw advU)
        + Db * df.inner(df.grad(CF), df.grad(test))  # **diffuse dissolved only**
        + test * AkVb * CF  # exchange source
    )


def funR(CFn, CTn, CFtn):
    ww = df.as_vector([0.5, 0.5])
    # use raw advU for consistency with Pw and weakL
    return -ww * (AkVb * CFtn - advU * df.dot(df.grad(CTn), v_dir) - AkVb * CFn)


# -----------------------------------------------------------------------------
# 0.  LINEARISED PRE-SOLVE  (assume CT ≈ CF so CB = 0)
# -----------------------------------------------------------------------------
U_lin = df.TrialFunction(V)
CF_lin, CFt_lin = df.split(U_lin)

# blood operator with CT=CF
Fb_lin = (weakL(phi_b, CF_lin, CF_lin) - AkVb * CFt_lin * phi_b) * df.dx

# tissue operator, drop nonlinear uptake term
Ft_lin = (
    -AkVt * (CF_lin - CFt_lin) * phi_t + Dt * df.inner(df.grad(CFt_lin), df.grad(phi_t))
) * df.dx

try:
    # Basic partition/dofs diagnostics
    try:
        ndofs_loc = V.dofmap().local_dimension()
    except Exception:
        ndofs_loc = -1
    print(f"R{rank}: Mesh cells={mesh.num_cells()}, dofs_local={ndofs_loc}, no_bc={no_bc}")
    print(f"R{rank}: Building linear warm-start forms...")
    sys.stdout.flush()
    a_lin = df.lhs(Fb_lin + Ft_lin)
    L_lin = df.rhs(Fb_lin + Ft_lin)
    solved = False
    if MPI.COMM_WORLD.Get_size() > 1:
        # In parallel, avoid df.solve default LU; use PETSc KSP explicitly
        print(f"R{rank}: Assembling linear system (parallel)...")
        sys.stdout.flush()
        A_lin, b_lin = df.assemble_system(a_lin, L_lin, bcs)
        print(f"R{rank}: Assembly done. Gathering diagnostics...")
        sys.stdout.flush()
        # Diagnostics: matrix/vector sizes and ownership
        try:
            gl_rows, gl_cols = A_lin.size(0), A_lin.size(1)
        except Exception:
            gl_rows = gl_cols = -1
        try:
            lsize_u = U_mixed.vector().local_size()
        except Exception:
            lsize_u = -1
        try:
            mat = A_lin.mat()
            r0, r1 = mat.getOwnershipRange()
            try:
                c0, c1 = mat.getOwnershipRangeColumn()
            except Exception:
                c0 = c1 = None
            try:
                info = mat.getInfo()
                nz_used = info.get("nz_used", None)
            except Exception:
                nz_used = None
            print(
                f"R{rank}: A_glob=({gl_rows},{gl_cols}), OwnRows=[{r0},{r1}), OwnCols={([c0,c1] if c0 is not None else None)}, vec_loc={lsize_u}, nz_used={nz_used}"
            )
            sys.stdout.flush()
        except Exception as _e_di:
            print(f"R{rank}: diag-matrix-info failed: {_e_di}")
            sys.stdout.flush()
        print(f"R{rank}: Starting KSP attempts...")
        sys.stdout.flush()
        # Try a few safe PETSc configurations in order (avoid hypre first)
        solver_choices = []
        if force_ksp and force_pc:
            solver_choices.append((force_ksp, force_pc))
        solver_choices += [
            ("gmres", "gamg"),
            ("gmres", "asm"),
            ("bicgstab", "ilu"),
        ]
        if not skip_hypre:
            solver_choices.append(("gmres", "hypre_amg"))
        for ksp_type, pc_type in solver_choices:
            try:
                print(f"R{rank}: Attempting KSP '{ksp_type}' with PC '{pc_type}'")
                sys.stdout.flush()
                ksp = df.KrylovSolver(ksp_type, pc_type)
                ksp.parameters["monitor_convergence"] = True
                ksp.parameters["report"] = True
                ksp.parameters["relative_tolerance"] = 1e-8
                ksp.parameters["absolute_tolerance"] = 1e-12
                ksp.parameters["maximum_iterations"] = 500
                print(f"R{rank}: Solving linear variational problem ({ksp_type}+{pc_type}).")
                sys.stdout.flush()
                ksp.solve(A_lin, U_mixed.vector(), b_lin)
                solved = True
                break
            except Exception as _e_cfg:
                if rank == 0:
                    print(f"KSP config failed ({ksp_type}+{pc_type}):", repr(_e_cfg))
                    sys.stdout.flush()
                continue
    else:
        # Serial: default solve is fine and fastest
        print("Rank 0: Solving linear variational problem (direct df.solve in serial).")
        df.solve(a_lin == L_lin, U_mixed, bcs)
        solved = True
    if not solved:
        raise RuntimeError("Linear warm-start not solved with any KSP config")
except Exception as e_lin:
    if rank == 0:
        print("Linear warm-start fallback path:", repr(e_lin))
    # Try LinearVariationalSolver with LU (serial) or KSP (parallel)
    try:
        problem_lin = df.LinearVariationalProblem(a_lin, L_lin, U_mixed, bcs)
        solver_lin = df.LinearVariationalSolver(problem_lin)
        if MPI.COMM_WORLD.Get_size() > 1:
            solver_lin.parameters["linear_solver"] = "gmres"
            solver_lin.parameters["preconditioner"] = "hypre_amg"
        else:
            solver_lin.parameters["linear_solver"] = "lu"
        solver_lin.solve()
    except Exception as e2:
        # Last resort: assemble and use LUSolver in serial, KSP in parallel
        A_lin, b_lin = df.assemble_system(a_lin, L_lin, bcs)
        if MPI.COMM_WORLD.Get_size() > 1:
            ksp = df.KrylovSolver("gmres", "hypre_amg")
            ksp.solve(A_lin, U_mixed.vector(), b_lin)
        else:
            lin_solver = df.LUSolver()
            lin_solver.solve(A_lin, U_mixed.vector(), b_lin)

CF_sol, CFt_sol = U_mixed.split()
F_res = (
    (weakL(phi_b, CF_sol, CF_sol) - AkVb * CFt_sol * phi_b)
    + (
        -AkVt * (CF_sol - CFt_sol) * phi_t
        + Dt * df.inner(df.grad(CFt_sol), df.grad(phi_t))
    )
) * df.dx
print("Linear warm-start ‖R‖ =", df.assemble(F_res).norm("l2"))

# -----------------------------------------------------------------------------
#  PSEUDO-TIME LOOP
# -----------------------------------------------------------------------------

maxG_val = 70e-12 / pO2C * Ghypertrophy  # [mol mm⁻³ s⁻¹]
num_steps = 10
pseudo_dt = df.Constant(1e2)

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
    ramp_factor = 0.0 if step == 0 else (step / float(num_steps))
    maxG = assign_local_property_vertexBased(mesh, maxG_val * ramp_factor, V0)

    # Ramp wall exchange as well (start tiny), recompute AkVb/AkVt inside loop
    kW_loop = assign_local_property_vertexBased(
        mesh, kWratioTmp * 35.0 * 0.001 * max(0.05, ramp_factor), V0
    )
    AkVb = df.project(
        df.conditional(
            df.gt(Vb, df.DOLFIN_EPS), kW_loop * Asurf / Vb, df.Constant(0.0)
        ),
        V_cg,
    )
    AkVt = df.project(
        df.conditional(
            df.gt(Vtis, df.DOLFIN_EPS), kW_loop * Asurf / Vtis, df.Constant(0.0)
        ),
        V_cg,
    )
    # Cap exchange rates to upper bound to limit stiffness
    Ak_cap = 1e2
    _akb = AkVb.vector().get_local()
    _akb = np.clip(_akb, 0.0, Ak_cap)
    AkVb.vector().set_local(_akb)
    AkVb.vector().apply("insert")
    _akt = AkVt.vector().get_local()
    _akt = np.clip(_akt, 0.0, Ak_cap)
    AkVt.vector().set_local(_akt)
    AkVt.vector().apply("insert")

    # Derived fields (avoid ufl.variable; clamp U to avoid negative fractional powers)
    U_pos = df.conditional(df.ge(U, df.Constant(0.0)), U, df.Constant(0.0))
    CB = 4 * CHb * Hct * SHb(mesh, U_pos, pO2C)
    CT = CB + U
    consumption = maxG * Ut / (Ut + km + df.Constant(1e-24))

    # Blood residual (incl. SUPG)
    Fb = (weakL(phi_b, U, CT) - AkVb * Ut * phi_b) * df.dx

    if steadySUPG:
        # Compute SUPG tau numerically to avoid UFL math on Functions
        _au = advU_safe.vector().get_local()
        _dl = dL.vector().get_local()
        dif = difD_value
        _pe = _au * _dl / (2.0 * (dif + dif / 65.0))
        _pe = np.maximum(_pe, 1e-12)
        _sigma = (_dl / (2.0 * _au)) * (1.0 / np.tanh(_pe) - 1.0 / _pe)
        tau_scalar = df.Function(V_cg)
        tau_scalar.vector().set_local(_sigma)
        tau_scalar.vector().apply("insert")
        tau = tau_scalar * W_inv
        # Fb += df.inner(tau*Pw, funR(U, CT, Ut))*df.dx
        Fb += df.dot(tau * Pw_vec, funR(U, CT, Ut)) * df.dx
    # Tissue residual – **sign fixed** (+AkVt)
    Ft = (
        (Ut - Ut_old) / pseudo_dt * phi_t
        - AkVt * (U - Ut) * phi_t
        + consumption * phi_t
        + Dt * df.inner(df.grad(Ut), df.grad(phi_t))
        + Dmb * CMb * df.inner(df.grad(Ut / (Ut + C50)), df.grad(phi_t))
    ) * df.dx

    F = Fb + Ft
    # Mild artificial diffusion on blood in first step for robustness
    if step == 0:
        F += (0.3 * Db) * df.inner(df.grad(U), df.grad(phi_b)) * df.dx
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

        current_dt = max(current_dt / 2.0, 1e-4)
        if rank == 0:
            print(
                f"Solve failed on try {try_id+1}; reducing dt to {current_dt:.4e} and retrying."
            )
        try_id += 1

    if try_id == max_retries and success_local == 0:
        if rank == 0:
            print(
                "Step failed after retries; stopping early. Last completed step is saved."
            )
        # Save the last successful checkpoint of previous step is already on disk
        break

    # Update pseudo-time variable
    _, Ut_new = U_mixed.split(deepcopy=True)
    Ut_old.assign(Ut_new)

    # Write results and checkpoint
    sid = step + 1
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
