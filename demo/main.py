import dolfin as df
import numpy as np
import scipy.io as sio
import os, sys, json
from datetime import datetime

# (No direct UFL utilities imported; keep imports minimal)
from mpi4py import MPI
from pathlib import Path

base_dir = "/workspace"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

"""
INITIAL SET-UP (quiet): initialise MPI and basic parameters without diagnostics.
"""
commMPI = MPI.COMM_WORLD
rank = commMPI.Get_rank()
sizeMPI = commMPI.Get_size()

# Quiet mode: remove verbose diagnostics and argv-based overrides

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
except Exception:
    pass

# Results directory handling: create a unique folder per run to avoid overwrite
def _init_results_dir(base_name: str = "results_1876v") -> str:
    """Create a unique results directory for this run.

    Rank 0 picks a timestamped name (optionally suffixed by env RESULTS_SUFFIX),
    ensures creation, then broadcasts the path to all ranks.
    """
    comm = MPI.COMM_WORLD
    run_dir = None
    if comm.rank == 0:
        # Policy:
        # - If base_name is exactly 'results_1876v': keep that folder name exactly
        #   (create if missing), do NOT append timestamp or index.
        # - Else (e.g., results_<suffix>): ensure uniqueness by appending _NN if exists.
        if base_name == "results_1876v":
            candidate = Path(".") / base_name
            candidate.mkdir(parents=True, exist_ok=True)
        else:
            base = base_name
            candidate = Path(".") / base
            idx = 0
            while candidate.exists():
                idx += 1
                candidate = Path(".") / f"{base}_{idx:02d}"
            candidate.mkdir(parents=True, exist_ok=False)
        run_dir = str(candidate)
        print(f"Results directory: {run_dir}")
    # Share the chosen directory with all ranks
    run_dir = comm.bcast(run_dir, root=0)
    # Non-root ranks ensure it exists (NOP if already created)
    if comm.rank != 0:
        os.makedirs(run_dir, exist_ok=True)
    comm.Barrier()
    return run_dir


# Lightweight CLI/env config
RESULTS_SUFFIX = os.environ.get("RESULTS_SUFFIX", "").strip()
OUTPUT_FORMAT = os.environ.get("OUTPUT_FORMAT", "both").strip().lower()
for arg in sys.argv[1:]:
    if arg.startswith("--results-suffix="):
        RESULTS_SUFFIX = arg.split("=", 1)[1].strip()
    elif arg.startswith("--output-format="):
        OUTPUT_FORMAT = arg.split("=", 1)[1].strip().lower()

# Normalise output selection
if OUTPUT_FORMAT not in ("pvd", "xdmf", "both"):
    OUTPUT_FORMAT = "both"

RESULTS_DIR = _init_results_dir("results_1876v" if not RESULTS_SUFFIX else f"results_{RESULTS_SUFFIX}")
CKPT_META = os.path.join(RESULTS_DIR, "checkpoint_meta.json")
CKPT_H5 = os.path.join(RESULTS_DIR, "checkpoint.h5")


def _check_finite(name, f):
    try:
        arr = f.vector().get_local()
        if not np.all(np.isfinite(arr)):
            if MPI.COMM_WORLD.rank == 0:
                print(f"[nan-check] {name} contains non-finite values: min={np.nanmin(arr)}, max={np.nanmax(arr)}")
            return False
    except Exception:
        pass
    return True


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
steadySUPG = 1  # enable SUPG stabilization
supg_enable_nsteps = 7  # only use SUPG in first few steps
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

# MPI-robust fill of DG0 fields from cell tags (owned dofs only)
dm = V_dg.dofmap()
r0, r1 = dm.ownership_range()
loc_Q = np.zeros(r1 - r0, dtype=float)
loc_R = np.zeros(r1 - r0, dtype=float)
for cell in df.cells(mesh):
    tag = int(cell_tags[cell]) - 1
    tag = 0 if tag < 0 else tag
    gd = dm.cell_dofs(cell.index())[0]
    if r0 <= gd < r1:
        loc_Q[gd - r0] = float(Qvessel[tag])
        loc_R[gd - r0] = float(Rvessel[tag])
Qcell.vector().set_local(loc_Q)
Rcell.vector().set_local(loc_R)
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
# Upper cap to limit extreme velocities (reduce stiffness) using a global 95th percentile
try:
    u95_loc = float(np.percentile(_au, 95)) if _au.size > 0 else 0.0
    umax = MPI.COMM_WORLD.allreduce(u95_loc, op=MPI.MAX)
    if not np.isfinite(umax) or umax <= df.DOLFIN_EPS:
        umax = 5.0
except Exception:
    umax = 5.0
_au[_au > umax] = umax
advU_safe.vector().set_local(_au)
advU_safe.vector().apply("insert")

# Debug sanity checks
_check_finite("Across_safe", Across_safe)
_check_finite("advU_safe", advU_safe)

# Vessel direction vectors (MPI-robust build per cell, then project to CG1)
from auxiliaryFunctions_dolfin import cell_tangent_dg_safe

v_dir_DG = cell_tangent_dg_safe(mesh)
v_dir = df.project(v_dir_DG, df.VectorFunctionSpace(mesh, "CG", 1))
if sizeMPI > 1:
    # In dolfin, finalize assembly and update ghosts via apply("insert")
    v_dir.vector().apply("insert")
    commMPI.barrier()
_check_finite("v_dir", v_dir)

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
bc_in = df.DirichletBC(V.sub(0), df.Constant(100), vertex_tags, INLET_TAG)
bc_out = df.DirichletBC(V.sub(0), df.Constant(20), vertex_tags, OUTLET_TAG)
bcs = [bc_in, bc_out]

# -----------------------------------------------------------------------------
#  SUPG MATRICES
# -----------------------------------------------------------------------------
W = df.as_matrix([[7 / 24, -1 / 24], [13 / 24, 5 / 24]])
W_inv = df.inv(W)
phi_grad = df.dot(df.grad(phi_b), v_dir)
# Use clamped advection (advU_safe) for stability
Pw = W * advU_safe * phi_grad

Pw_vec = df.as_vector(
    [advU_safe * phi_grad, advU_safe * phi_grad]
)


# -----------------------------------------------------------------------------
#  HELPER: weak form of blood operator
# -----------------------------------------------------------------------------


def weakL(test, CF, CT, ADV):
    """Return weak form of blood equation (no SUPG).
    Diffusion applies *only* to dissolved O₂ (CF)."""
    return (
        test * ADV * df.dot(df.grad(CT), v_dir)  # advection of total O₂ (possibly off)
        + Db * df.inner(df.grad(CF), df.grad(test))  # **diffuse dissolved only**
        + test * AkVb * CF  # exchange source
    )


def funR(CFn, CTn, CFtn, ADV):
    ww = df.as_vector([0.5, 0.5])
    # use provided ADV for consistency with Pw and weakL
    return -ww * (AkVb * CFtn - ADV * df.dot(df.grad(CTn), v_dir) - AkVb * CFn)


# -----------------------------------------------------------------------------
# 0.  LINEARISED PRE-SOLVE  (assume CT ≈ CF so CB = 0)
# -----------------------------------------------------------------------------
U_lin = df.TrialFunction(V)
CF_lin, CFt_lin = df.split(U_lin)

# blood operator with CT=CF
Fb_lin = (weakL(phi_b, CF_lin, CF_lin, advU_safe) - AkVb * CFt_lin * phi_b) * df.dx

# tissue operator, drop nonlinear uptake term
Ft_lin = (
    -AkVt * (CF_lin - CFt_lin) * phi_t + Dt * df.inner(df.grad(CFt_lin), df.grad(phi_t))
) * df.dx

try:
    a_lin = df.lhs(Fb_lin + Ft_lin)
    L_lin = df.rhs(Fb_lin + Ft_lin)
    solved = False
    if MPI.COMM_WORLD.Get_size() > 1:
        # In parallel, avoid df.solve default LU; use PETSc KSP explicitly
        A_lin, b_lin = df.assemble_system(a_lin, L_lin, bcs)
        # Assemble done; attempt KSP solves
        # Try a few safe PETSc configurations in order (avoid hypre first)
        solver_choices = [
            ("gmres", "gamg"),
            ("gmres", "asm"),
            ("bicgstab", "ilu"),
            ("gmres", "hypre_amg"),
        ]
        for ksp_type, pc_type in solver_choices:
            try:
                ksp = df.KrylovSolver(ksp_type, pc_type)
                ksp.parameters["relative_tolerance"] = 1e-8
                ksp.parameters["absolute_tolerance"] = 1e-12
                ksp.parameters["maximum_iterations"] = 500
                ksp.solve(A_lin, U_mixed.vector(), b_lin)
                solved = True
                break
            except Exception:
                continue
    else:
        # Serial: default solve is fine and fastest
        df.solve(a_lin == L_lin, U_mixed, bcs)
        solved = True
    if not solved:
        raise RuntimeError("Linear warm-start not solved with any KSP config")
except Exception as e_lin:
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

# Warm-start complete

# -----------------------------------------------------------------------------
#  PSEUDO-TIME LOOP
# -----------------------------------------------------------------------------

maxG_val = 70e-12 / pO2C * Ghypertrophy  # [mol mm⁻³ s⁻¹]
num_steps = 10000
# Output control: write XDMF results every N steps
SAVE_EVERY_N_STEPS = 50
# Start with smaller pseudo-time step and allow adaptive growth (conservative)
pseudo_dt = df.Constant(1e-2)
DT_GROWTH = 1.5
DT_MAX = 1.0

# Adaptive dt growth controls
DT_GROWTH_BASE = 1.2
DT_GROWTH_FAST = 1.4
DT_GROWTH_SLOW = 1.05
CFL_ADV = 0.4  # CFL-like cap using advective speed

# Optional early termination on steady update
STEADY_STOP_ENABLE = True
STEADY_TOL = 1e-4          # relative update threshold
STEADY_WINDOW = 2          # consecutive steps within tol
MIN_STEPS_BEFORE_STEADY = 3  # wait a few steps before checking
steady_hits = 0

# Output format selection and writer prep
want_pvd = OUTPUT_FORMAT in ("pvd", "both")
want_xdmf = OUTPUT_FORMAT in ("xdmf", "both")

# Prepare PVD time-series writers
try:
    vtkfile_b = df.File(os.path.join(RESULTS_DIR, "CFb_timeseries.pvd")) if want_pvd else None
    vtkfile_t = df.File(os.path.join(RESULTS_DIR, "CFt_timeseries.pvd")) if want_pvd else None
except Exception:
    vtkfile_b = None
    vtkfile_t = None

# Prepare XDMF time-series writers
try:
    xdmf_b = (
        df.XDMFFile(commMPI, os.path.join(RESULTS_DIR, "CFb_timeseries.xdmf")) if want_xdmf else None
    )
    xdmf_t = (
        df.XDMFFile(commMPI, os.path.join(RESULTS_DIR, "CFt_timeseries.xdmf")) if want_xdmf else None
    )
    if xdmf_b is not None:
        xdmf_b.parameters["flush_output"] = True
        xdmf_b.parameters["functions_share_mesh"] = True
    if xdmf_t is not None:
        xdmf_t.parameters["flush_output"] = True
        xdmf_t.parameters["functions_share_mesh"] = True
except Exception:
    xdmf_b = None
    xdmf_t = None

# Accumulated pseudo-time for time-series outputs
t_accum = 0.0

# Additional continuation controls to ease early Newton solves
# Ramp controls (more conservative in MPI)
RAMP_ADV_STEPS = 8 if sizeMPI > 1 else 3        # ramp advection over first steps
RAMP_HCT_ONSET = 4 if sizeMPI > 1 else 2        # keep Hct=0 longer in MPI
DMB_ONSET_STEP = 5 if sizeMPI > 1 else 3        # enable membrane diffusion later in MPI
EXTRA_DIFF_STEPS = 5 if sizeMPI > 1 else 3      # add decaying artificial diffusion for first steps

# Prepare previous tissue state and resume support
_, Ut_old = U_mixed.split(deepcopy=True)
USE_RESUME = False
if USE_RESUME:
    resume_found, last_step_done, ckpt_dt = load_checkpoint(U_mixed)
else:
    resume_found, last_step_done, ckpt_dt = (False, -1, None)
start_step = 0
if resume_found:
    start_step = last_step_done + 1
    if ckpt_dt is not None:
        try:
            # Clamp loaded dt into sane bounds
            _dt_loaded = float(ckpt_dt)
            if _dt_loaded <= 0:
                _dt_loaded = 1e-3
            _dt_loaded = min(_dt_loaded, DT_MAX)
            pseudo_dt.assign(_dt_loaded)
        except Exception:
            _dt_loaded = min(max(float(ckpt_dt), 1e-3), DT_MAX)
            pseudo_dt.assign(df.Constant(_dt_loaded))
    # Keep pseudo-time state consistent on resume
    try:
        _, _Ut_resume = U_mixed.split(deepcopy=True)
        Ut_old.assign(_Ut_resume)
    except Exception:
        pass
    if MPI.COMM_WORLD.rank == 0:
        try:
            print(f"[resume] last step={last_step_done}, start_step={start_step}, dt={float(pseudo_dt.values()[0]):g}")
        except Exception:
            pass

for step in range(start_step, num_steps):

    # Ramp metabolism: zero on first step, then gradual
    ramp_factor = 0.0 if step == 0 else (step / float(num_steps))
    maxG = assign_local_property_vertexBased(mesh, maxG_val * ramp_factor, V0)

    # Ramp wall exchange as well (start tiny), recompute AkVb/AkVt inside loop
    kW_loop = assign_local_property_vertexBased(
        mesh, kWratioTmp * 35.0 * 0.001 * max(0.0, ramp_factor), V0
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
    Ak_cap = 5.0 if step == 0 else 1e1
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
    # Ramp haematocrit: keep off for first few steps, then follow global ramp
    hct_ramp = 0.0 if step < RAMP_HCT_ONSET else float(ramp_factor)
    Hct_eff = df.Constant(HctTmp * hct_ramp)
    CB = 4 * CHb * Hct_eff * SHb(mesh, U_pos, pO2C)
    CT = CB + U
    # Clamp tissue concentration for nonlinear terms to avoid negative-induced instabilities
    Ut_pos = df.conditional(df.ge(Ut, df.Constant(0.0)), Ut, df.Constant(0.0))
    consumption = maxG * Ut_pos / (Ut_pos + km + df.Constant(1e-24))

    # Sanity checks to catch NaNs early
    _check_finite("AkVb", AkVb)
    _check_finite("AkVt", AkVt)
    try:
        Ublood, Utissue = U_mixed.split()
        _check_finite("U", Ublood)
        _check_finite("Ut", Utissue)
    except Exception:
        pass

    # Optional Picard bootstrap in early steps: decouple and pre-smooth U/Ut
    if step < 2:
        try:
            # Local copies on separate spaces
            Ub_iter, Ut_iter = U_mixed.split(deepcopy=True)
            # Prepare V0/V1 boundary conditions for blood
            bc_in0 = df.DirichletBC(V0, df.Constant(100), vertex_tags, INLET_TAG)
            bc_out0 = df.DirichletBC(V0, df.Constant(20), vertex_tags, OUTLET_TAG)
            bcs0 = [bc_in0, bc_out0]

            pb = df.TestFunction(V0)
            CF = df.TrialFunction(V0)
            vt_lin = df.TestFunction(V1)
            CFt = df.TrialFunction(V1)

            picard_iters = 3
            for _ in range(picard_iters):
                # Blood linearized with Ut frozen
                Fb_pic = (
                    pb * advU_safe * df.dot(df.grad(CF), v_dir)
                    + Db * df.inner(df.grad(CF), df.grad(pb))
                    + pb * AkVb * CF
                    - AkVb * Ut_iter * pb
                ) * df.dx
                a_b, L_b = df.lhs(Fb_pic), df.rhs(Fb_pic)
                df.solve(a_b == L_b, Ub_iter, bcs0)

                # Tissue linearized with U frozen; ignore Dmb and consumption in early steps
                Ft_pic = (
                    (CFt - Ut_old) / pseudo_dt * vt_lin
                    - AkVt * (Ub_iter - CFt) * vt_lin
                    + Dt * df.inner(df.grad(CFt), df.grad(vt_lin))
                ) * df.dx
                a_t, L_t = df.lhs(Ft_pic), df.rhs(Ft_pic)
                df.solve(a_t == L_t, Ut_iter)

            # Assign back to mixed
            df.FunctionAssigner(V, [V0, V1]).assign(U_mixed, [Ub_iter, Ut_iter])
            # Refresh derived quantities after bootstrap
            U_pos = df.conditional(df.ge(U, df.Constant(0.0)), U, df.Constant(0.0))
            Hct_eff = df.Constant(HctTmp * ramp_factor)
            CB = 4 * CHb * Hct_eff * SHb(mesh, U_pos, pO2C)
            CT = CB + U
        except Exception:
            pass

    # Optional Picard mixed linearization in first step for a better initial guess
    if step == 0:
        try:
            picard_iters = 10
            relax_pic = 0.7
            # Local copies for iterates
            Ub_it, Ut_it = U_mixed.split(deepcopy=True)
            Wt = df.TrialFunction(V)
            CF_new, CFt_new = df.split(Wt)
            ADV_used = df.Constant(0.0)
            Dmb_eff = df.Constant(0.0)
            for k_it in range(picard_iters):
                # Build lagged nonlinear pieces
                U_it_pos = df.conditional(df.ge(Ub_it, df.Constant(0.0)), Ub_it, df.Constant(0.0))
                Ut_it_pos = df.conditional(df.ge(Ut_it, df.Constant(0.0)), Ut_it, df.Constant(0.0))
                CB_it = 4 * CHb * df.Constant(HctTmp * ramp_factor) * SHb(mesh, U_it_pos, pO2C)
                CT_it = CB_it + Ub_it
                cons_it = maxG * Ut_it_pos / (Ut_it_pos + km + df.Constant(1e-24))

                Fb_pic = (
                    phi_b * ADV_used * df.dot(df.grad(CT_it), v_dir)
                    + Db * df.inner(df.grad(CF_new), df.grad(phi_b))
                    + phi_b * AkVb * CF_new
                    - AkVb * CFt_new * phi_b
                ) * df.dx
                Ft_pic = (
                    (CFt_new - Ut_old) / pseudo_dt * phi_t
                    - AkVt * (CF_new - CFt_new) * phi_t
                    + cons_it * phi_t
                    + Dt * df.inner(df.grad(CFt_new), df.grad(phi_t))
                    + Dmb_eff * CMb * df.inner(
                        df.grad(Ut_it_pos / (Ut_it_pos + C50)), df.grad(phi_t)
                    )
                ) * df.dx
                a_pic = df.lhs(Fb_pic + Ft_pic)
                L_pic = df.rhs(Fb_pic + Ft_pic)
                U_tmp = df.Function(V)
                df.solve(a_pic == L_pic, U_tmp, bcs)
                # Relaxed update of iterates
                Ub_new, Ut_new_lin = U_tmp.split()
                Ub_it.vector().axpy(relax_pic, Ub_new.vector())
                Ub_it.vector().axpby(1.0 - relax_pic, 0.0, Ub_it.vector())
                Ut_it.vector().axpy(relax_pic, Ut_new_lin.vector())
                Ut_it.vector().axpby(1.0 - relax_pic, 0.0, Ut_it.vector())
                Ub_it.vector().apply("insert")
                Ut_it.vector().apply("insert")
                # Simple stopping by update norm
                try:
                    diff = (U_tmp.vector() - U_mixed.vector()).norm("l2")
                    ref = max(1e-12, U_mixed.vector().norm("l2"))
                    if MPI.COMM_WORLD.rank == 0:
                        print(f"[step 0 picard] iter {k_it+1}/{picard_iters} update rel={diff/ref:.3e}")
                    if diff / ref < 1e-3:
                        break
                except Exception:
                    pass
            # Assign iterate back
            df.FunctionAssigner(V, [V0, V1]).assign(U_mixed, [Ub_it, Ut_it])
        except Exception:
            pass

    # Blood residual (incl. SUPG) with advection ramped in over first steps
    adv_factor = 0.0 if step == 0 else min(1.0, float(step) / float(RAMP_ADV_STEPS))
    ADV_used = advU_safe * df.Constant(adv_factor)
    Fb = (weakL(phi_b, U, CT, ADV_used) - AkVb * Ut * phi_b) * df.dx

    # Use SUPG during early steps to stabilize advection
    use_supg = bool(steadySUPG) and (step > 0) and (step < supg_enable_nsteps)
    if use_supg:
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
        Fb += df.dot(tau * Pw_vec, funR(U, CT, Ut, ADV_used)) * df.dx
    # Tissue residual – **sign fixed** (+AkVt)
    Ft = (
        (Ut - Ut_old) / pseudo_dt * phi_t
        - AkVt * (U - Ut) * phi_t
        + consumption * phi_t
        + Dt * df.inner(df.grad(Ut), df.grad(phi_t))
        + (df.Constant(0.0) if step < DMB_ONSET_STEP else Dmb) * CMb
        * df.inner(df.grad(Ut_pos / (Ut_pos + C50)), df.grad(phi_t))
    ) * df.dx

    F = Fb + Ft
    # Mild artificial diffusion on blood in early steps for robustness (decays to 0)
    if step < EXTRA_DIFF_STEPS:
        decay = float(EXTRA_DIFF_STEPS - step) / float(EXTRA_DIFF_STEPS)
        F += (0.5 * decay * Db) * df.inner(df.grad(U), df.grad(phi_b)) * df.dx
    J = df.derivative(F, U_mixed, δ)

    # Newton solve with DOLFIN's NonlinearVariationalSolver
    # Optional assembly checks removed for cleanliness

    problem = df.NonlinearVariationalProblem(F, U_mixed, bcs, J)
    solver = df.NonlinearVariationalSolver(problem)
    prm = solver.parameters
    # Set Newton tolerances based on global dof count (looser), and use faster line search
    try:
        ndofs = V.dim()
        # Absolute tol scaled with sqrt(N) so it is not unrealistically small
        abs_tol = max(1e-8, 1e-6 * float(ndofs) ** 0.5)
        prm["newton_solver"]["relative_tolerance"] = 2e-3
        prm["newton_solver"]["absolute_tolerance"] = abs_tol
        prm["newton_solver"]["maximum_iterations"] = 60
        prm["newton_solver"]["linear_solver"] = "gmres" if MPI.COMM_WORLD.Get_size() > 1 else "lu"
        # Use faster backtracking line search once continuation is in place
        prm["newton_solver"]["line_search"] = "bt"
        # Choose preconditioner appropriate to MPI size
        # In MPI, rely on PETScOptions (pc_type=gamg set above); do not force hypre
        if MPI.COMM_WORLD.Get_size() == 1:
            prm["newton_solver"]["preconditioner"] = "ilu"
        prm["newton_solver"]["error_on_nonconvergence"] = False
        # Optional: reduce verbosity
        prm["newton_solver"]["report"] = False
        if MPI.COMM_WORLD.rank == 0:
            print(f"[newton] rel_tol=2e-3 abs_tol={abs_tol:.3e} (ndofs={ndofs})")
    except Exception:
        pass

    # Retry with pseudo_dt backoff if failure occurs
    max_retries = 8
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
            # Accept solve if vector is finite (treat as success);
            # some dolfin builds don't expose a reliable converged() API
            vec = U_mixed.vector().get_local()
            success_local = 1 if np.all(np.isfinite(vec)) else 0
        except Exception as e:
            success_local = 0

        # MPI agreement: all ranks must succeed
        success_global = MPI.COMM_WORLD.allreduce(success_local, op=MPI.MIN)
        if success_global == 1:
            if MPI.COMM_WORLD.rank == 0:
                print(f"[step {step}] Accepted solve at dt={current_dt:g}")
            break

        # Revert and back off dt, clamp negatives
        try:
            U_mixed.assign(U_prev)
        except Exception:
            df.FunctionAssigner(V, [V0, V1]).assign(U_mixed, list(U_prev.split()))
        vloc = U_mixed.vector().get_local()
        vloc[~np.isfinite(vloc)] = 0.0
        vloc[vloc < 0.0] = 0.0
        U_mixed.vector().set_local(vloc)
        U_mixed.vector().apply("insert")

        current_dt = max(current_dt / 2.0, 1e-6)
        if MPI.COMM_WORLD.rank == 0:
            print(f"[step {step}] Newton failed; backing off dt to {current_dt:g} (try {try_id+1}/{max_retries})")
        try_id += 1

    if try_id == max_retries and success_local == 0:
        if MPI.COMM_WORLD.rank == 0:
            print(f"[step {step}] Aborting pseudo-time loop: max retries exhausted.")
        # Save the last successful checkpoint of previous step is already on disk
        break

    # Update pseudo-time variable
    _, Ut_new = U_mixed.split(deepcopy=True)
    Ut_old.assign(Ut_new)

    # Advance accumulated pseudo-time by accepted dt for this step
    try:
        t_accum += float(current_dt)
    except Exception:
        pass

    # Write results (every SAVE_EVERY_N_STEPS) and checkpoint
    sid = step + 1
    if (sid % SAVE_EVERY_N_STEPS) == 0:
        Ublood, Utissue = U_mixed.split()
        Ublood.rename("CFb", "")
        Utissue.rename("CFt", "")
        # Append to requested outputs
        try:
            if want_pvd and (vtkfile_b is not None) and (vtkfile_t is not None):
                vtkfile_b << (Ublood, t_accum)
                vtkfile_t << (Utissue, t_accum)
        except Exception:
            pass
        try:
            if want_xdmf and (xdmf_b is not None) and (xdmf_t is not None):
                xdmf_b.write(Ublood, t_accum)
                xdmf_t.write(Utissue, t_accum)
        except Exception:
            pass

    # Persist checkpoint for resume (0-based step index)
    save_checkpoint(U_mixed, step, current_dt)

    # Increase pseudo-time step for next iteration on success (conservative & capped)
    try:
        # Measure update size to adapt growth
        try:
            upd = (U_mixed.vector() - U_prev.vector()).norm("l2")
            base = max(1e-12, U_mixed.vector().norm("l2"))
            upd_rel = upd / base
        except Exception:
            upd_rel = None

        growth = DT_GROWTH_BASE
        # If we had retries this step, be extra conservative
        if try_id > 0:
            growth = DT_GROWTH_SLOW
        if upd_rel is not None:
            if upd_rel > 0.2:
                growth = DT_GROWTH_SLOW
            elif upd_rel < 0.05:
                growth = DT_GROWTH_FAST

        # CFL-like cap based on advection and element size
        try:
            _a = advU_safe.vector().get_local()
            _d = dL.vector().get_local()
            if _a.size > 0:
                u95_loc = float(np.percentile(_a, 95))
            else:
                u95_loc = 0.0
            if _d.size > 0:
                h5_loc = float(np.percentile(_d, 5))
            else:
                h5_loc = 1e20
            u95 = MPI.COMM_WORLD.allreduce(u95_loc, op=MPI.MAX)
            h5 = MPI.COMM_WORLD.allreduce(h5_loc, op=MPI.MIN)
            dt_cfl = CFL_ADV * h5 / max(u95, 1e-12)
        except Exception:
            dt_cfl = DT_MAX

        # Model continuation-based cap: keep dt modest until all ramps are on
        if step < DMB_ONSET_STEP + 1:
            dt_model_cap = 0.03
        elif step < RAMP_HCT_ONSET + 2:
            dt_model_cap = 0.06
        else:
            dt_model_cap = 0.1

        dt_grown = current_dt * growth
        # Do not shrink dt on a successful step; only cap growth
        next_dt = min(dt_grown, max(current_dt, dt_cfl), max(current_dt, dt_model_cap), DT_MAX)
        pseudo_dt.assign(next_dt)
        if MPI.COMM_WORLD.rank == 0:
            cap_info = f"(growth={growth:.2f}, cfl_q={dt_cfl:.4g}, model_cap={dt_model_cap})"
            print(f"[step {step}] Increasing dt to {next_dt:g} for next step {cap_info}")
    except Exception:
        pass

    # Optional steady-state early exit
    if STEADY_STOP_ENABLE and (step + 1) >= MIN_STEPS_BEFORE_STEADY:
        try:
            upd = (U_mixed.vector() - U_prev.vector()).norm("l2")
            base = max(1e-12, U_mixed.vector().norm("l2"))
            rel_upd = upd / base
            if MPI.COMM_WORLD.rank == 0:
                print(f"[steady-check] rel_update={rel_upd:.3e} tol={STEADY_TOL:g} window={STEADY_WINDOW}")
            if rel_upd < STEADY_TOL:
                steady_hits += 1
            else:
                steady_hits = 0
            if steady_hits >= STEADY_WINDOW:
                if MPI.COMM_WORLD.rank == 0:
                    print(f"[steady] Reached steady tolerance {STEADY_TOL:g} for {STEADY_WINDOW} steps. Finishing.")
                break
        except Exception:
            pass

# Finished
try:
    if 'vtkfile_b' in locals() and vtkfile_b is not None:
        vtkfile_b.close()
    if 'vtkfile_t' in locals() and vtkfile_t is not None:
        vtkfile_t.close()
    if 'xdmf_b' in locals() and xdmf_b is not None:
        xdmf_b.close()
    if 'xdmf_t' in locals() and xdmf_t is not None:
        xdmf_t.close()
except Exception:
    pass
