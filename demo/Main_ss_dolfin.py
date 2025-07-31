from dolfin import *
import numpy as np
import scipy.io as sio
import os
import sys
from time import time
from ufl import tanh, dot, grad, inner, variable, max_value

from mpi4py import MPI
sys.path.append('../src')  # Adjust this path as necessary
from auxiliaryFunctions_dolfin import *
commMPI = MPI.COMM_WORLD
rank = commMPI.Get_rank()
sizeMPI = commMPI.Get_size()
# Start timer
start_time = time()

# Constants and conversion factors
Molar_to_mol_mm = 1E-6
mL_to_mm = 1000.0
mmHg_to_mmGs = 133.322
pO2C = 1.35E-12
constPO2inlet = 100
constQtmp = 3.9E-9 * mL_to_mm
constRtmp = 3.0 / 1000
constQFlag=0
constRFlag=0

useCTCFCB = 2
difD_value = 2.41e-5 * 100  ## Free O2 diffusion coefficient (cm^2/s=100mm^2/s): [0.95, 1.5, 2.1, 2.41]e-5cm^2/s
PeCritical = 1  # Use SUPG when 'Pe > PeCritical'!
steadySUPG =1
Ghypertrophy = 1.0
ratioVtVb = 12.5  
kWratioTmp = 1.0 #*5/3.5
HctTmp = 0.25
assign_local_Kw_Gmax = 1  # TODO: '0'=nonzero Kw and Gmax for ArtVen!!!

# Load mesh
mesh = Mesh()
with XDMFFile(commMPI, "../mesh/1876v_90TV_dL0.001_2tags.xdmf") as infile:
    infile.read(mesh)
    mvc_cells = MeshValueCollection("size_t", mesh, mesh.topology().dim())
    infile.read(mvc_cells, "Cell tags")
    cell_tags = cpp.mesh.MeshFunctionSizet(mesh, mvc_cells)
    mvc_vertices = MeshValueCollection("size_t", mesh, 0)
    infile.read(mvc_vertices, "mesh_tags")
    vertex_tags = MeshFunction("size_t", mesh, mvc_vertices)

# Access index-value pairs using the original MeshValueCollection (if needed)
cell_indices=[cell.index() for cell in cells(mesh)]
cell_values=[cell_tags[cell] for cell in cells(mesh)]

vertex_indices=[]
vertex_values=[]
for facet in facets(mesh):
    for vertex in vertices(facet):
        vertex_indices.append(vertex.index())
        vertex_values.append(vertex_tags[facet])
        
# Define tag values for inlets and outlets
inlet_tag_value = 1 # TODO
outlet_tag_value = 2  # TODO: [2, 4]
########

P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
element = MixedElement([P1, P1])
V = FunctionSpace(mesh, element)

V0 = V.sub(0).collapse()
V1 = V.sub(1).collapse()

dudut = TrialFunction(V)
vvt = TestFunction(V)
uut = Function(V)

du, dut = split(dudut)
v, vt = split(vvt)
u, ut = split(uut)
# Use non-negative versions for all nonlinear terms
u_safe = conditional(gt(u, Constant(0.0)), u, Constant(0.0))
ut_safe = conditional(gt(ut, Constant(0.0)), ut, Constant(0.0))

direction_vector_cells = compute_directional_vectors_cells(mesh)
dirVector_DG = cellDirVec_DG(mesh, direction_vector_cells)

target_vectorCGSpace = VectorFunctionSpace(mesh, "CG", 1)
dirVector = project(dirVector_DG, target_vectorCGSpace)
if commMPI.Get_size() > 1:
    dirVector.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  
    commMPI.barrier()  


matdata = sio.loadmat("../matlabData/1786v_90TV/DataSaved.mat")
Qvessel_mat = matdata['Qio']*1000   #[mL/s] --> [mm^3/s]
Qvessel_array_all = np.asfarray(Qvessel_mat, float)
Qvessel_array = Qvessel_array_all[-1:]  # Take only [-BCL:END][vID]

Rvessel_mat = matdata['Rat'] 
Rvessel_array_all = np.asfarray(Rvessel_mat, float)
Rvessel_array = Rvessel_array_all[-1:]

vLenMatlab = sio.loadmat("../matlabData/DataSaved_vID_Ord_Len.mat")
vLen_list = vLenMatlab['vLen'] / 1000.0

QvesselMatlab = Qvessel_array[0]
RvesselMatlab = Rvessel_array[0]

V_dg = FunctionSpace(mesh, "DG", 0)

# Initialize DG functions for Q and R
Qcell_function = Function(V_dg)
Rcell_function = Function(V_dg)

# Assign cell-wise values
Q_array = Qcell_function.vector().get_local()
R_array = Rcell_function.vector().get_local()

# Assign based on cell_tags (use same order from FEniCSx)
QvesselFEM = np.array([QvesselMatlab[int(vID - 1)] for cID, vID in zip(cell_indices, cell_values)])
RvesselFEM = np.array([RvesselMatlab[int(vID - 1)] for cID, vID in zip(cell_indices, cell_values)])

# Assign values to the local vector
Q_array[:] = QvesselFEM
R_array[:] = RvesselFEM

Qcell_function.vector().set_local(Q_array)
Qcell_function.vector().apply("insert")
Rcell_function.vector().set_local(R_array)
Rcell_function.vector().apply("insert")

# Optional: Sync for MPI (not needed for serial runs)
Qcell_function.vector().apply("insert")
Rcell_function.vector().apply("insert")


# Project DG0 to CG1
V_cg = FunctionSpace(mesh, "CG", 1)

Qnode_CG = project(Qcell_function, V_cg)
Rnode_CG = project(Rcell_function, V_cg)

dL_cell = CellDiameter(mesh)  # [mm]
dL = project(dL_cell, V_cg)


###### Better to compute Area and Volume from cells and then map to vertices!!!
Asurface_cell = 2.0 * np.pi * Rcell_function * dL_cell  # Surface area!!!
AcrossSect_cell = np.pi * Rcell_function ** 2  # Cross-sectional area!!!
Vb_cell = AcrossSect_cell * dL_cell  # Volume of blood or plasma
Asurface = project(Asurface_cell, V_cg)
AcrossSect = project(AcrossSect_cell, V_cg)
Vb = project(Vb_cell, V_cg)

CHb = Constant(5.3e-9)
Hct = Constant(HctTmp)
kWtmp = kWratioTmp * 35.0 * 0.001  # [mm/s]
kW = assign_local_property_vertexBased(mesh, kWtmp, V0)

Vtis = Vb * ratioVtVb
AkVt = project(kW * Asurface_cell / (Vb_cell* ratioVtVb), V_cg)  # kW * Asurface / Vtis
AkVb = project(kW * Asurface_cell / Vb_cell, V_cg)  # kW * Asurface / Vb


advU = project(Qcell_function / AcrossSect_cell, V_cg)

difD = difD_value
difDHb = difD / 65.0

#initialize variables
init_value_CFb = 100 * pO2C
init_value_CFt = 50 * pO2C

#init_CFb = interpolate(Constant(init_value_CFb), V0)
init_CFb = assign_initial_condition_vertex_based(mesh, V0, init_value_CFb)
init_CFt = interpolate(Constant(init_value_CFt), V1)

# Step 3: Assign to mixed function space
assigner = FunctionAssigner(V, [V0, V1])
assigner.assign(uut, [init_CFb, init_CFt])


#BCs
bc0_inlet = DirichletBC(V.sub(0), Constant(init_value_CFb), vertex_tags, inlet_tag_value)
bc0_outlet = DirichletBC(V.sub(0), Constant(20.0 * pO2C), vertex_tags, outlet_tag_value)

bcs = [bc0_inlet]+ [bc0_outlet]


W = as_matrix([[7.0 / 24.0, -1.0 / 24.0], [13.0 / 24.0, 5.0 / 24.0]])
W_inv = inv(W)
ww = as_vector([0.5, 0.5])
vv = as_vector([v, v])
Pw = W * advU * dot(grad(v), dirVector)
######


def weakL(v, CF, CT):
    return v * advU * dot(grad(CT), dirVector) \
         + difD * inner(grad(CT), grad(v)) + v * AkVb * CF

def funR(CFn, CTn, CFtn):
    return - ww * ( AkVb * CFtn - advU * dot(grad(CTn), dirVector) - AkVb * CFn )



# Optional: compute PeMax using numpy arrays (after evaluating functions)
U_array = advU.vector().get_local()
dL_array = dL.vector().get_local()
Pe_array = np.abs(U_array * dL_array) / (2 * float(difD))
PeMax = np.max(Pe_array)
print(f" PeMax = {PeMax}")


maxGvalue =70.0 * 1E-12 * Ghypertrophy  # [mol/(mm^3 s)]-->[uM/s]; 1uM=1e-12mol/mm^3
#maxG = assign_local_property_vertexBased(mesh, maxGvalue, V0)  

# Constants for tissue
#maxG = Constant( 0.01 * 70.0 * 1E-12)
km = Constant( 1e-7 * 1e-6)
Dmb = Constant( 2.2e-7 * 100)
CMb = Constant( 1E-4 * 1E-6)
C50 = Constant( 2.5 * mmHg_to_mmGs)
# Smooth approximation of max(ut, 0)



# Solve

# First to solve linear terms only- so weakL(v,u=CF,u=CF) is used instead of weakL(v, CF, CT)  as CT=CF+CB, and CB is nonlinearly related
Ft = - AkVt * (u - ut) * vt * dx + difD * inner(grad(ut), grad(vt)) * dx 
Fb = (ww[0] * weakL(v, u, u) + ww[1] * weakL(v, u, u)) * dx - (ww[0] * AkVb * ut * v + ww[1] * AkVb * ut * v) * dx

F = Fb + Ft
J = derivative(F, uut, dudut)
res = assemble(F)
print("Residual norm:", res.norm("l2"))
print("Residual min/max:", res.min(), res.max())


# --- SNES nonlinear solver setup ---
PETScOptions.set("snes_monitor")
PETScOptions.set("snes_linesearch_type", "bt")            # backtracking line search
PETScOptions.set("snes_linesearch_monitor", "")           # to check damping
PETScOptions.set("snes_linesearch_damping", 0.8)           # or lower if needed

problem = NonlinearVariationalProblem(F, uut, bcs, J)
solver = NonlinearVariationalSolver(problem)
solver.parameters["nonlinear_solver"] = "snes"
solver.parameters["snes_solver"]["report"] = True
solver.parameters["snes_solver"]["absolute_tolerance"] = 1e-13
solver.parameters["snes_solver"]["maximum_iterations"] = 1000
solver.parameters["snes_solver"]["linear_solver"] = "lu"  # or "cg", "gmres", etc.
# Optionally: solver.parameters["snes_solver"]["preconditioner"] = "ilu"
solver.solve()

# Saving linear solution
xdmf_out = XDMFFile(commMPI, f"./results_1876v/CFb_linear.xdmf")
u0, u1 = uut.split()
u0.rename("CFb", "")
xdmf_out.write(u0)
xdmf_out.close()
xdmf_out1 = XDMFFile(commMPI, f"./results_1876v/CFt_linear.xdmf")
u1.rename("CFt", "")
xdmf_out1.write(u1)
xdmf_out1.close()

# --- Pseudo-time stepping setup ---
num_timesteps = 10
dtau = Constant(1e2)  # pseudo-time step size

# Create function to hold previous ut

# Split with deepcopy to ensure independence
_, ut_old = uut.split(deepcopy=True)

for timestep in range(num_timesteps):
    print(f"\n=== Pseudo Time Step {timestep+1}/{num_timesteps} ===")

    # Ramp maxG within time stepping
    ramp = (timestep + 1) / num_timesteps
    #ramp=0.5
    maxG_ramped = assign_local_property_vertexBased(mesh, maxGvalue * ramp, V0)
    if timestep > 0:
        PETScOptions.set("snes_linesearch_damping", 0.5)           # or lower if needed

    # Consumption using current ut
    ut_safe = conditional(gt(ut, Constant(0.0)), ut, Constant(0.0))
    u_safe = conditional(gt(u, Constant(0.0)), u, Constant(0.0))

    consumption = maxG_ramped * ut_safe / (ut_safe + km + Constant(1e-24))

    # Diagnostic: print residual norm before solve
    res_timestep = assemble(F)
    print(f"[Diagnostic] Residual norm before solve: {res_timestep.norm('l2'):.3e}")

    # Artificial time term
    Ft_time = ((ut - ut_old) / dtau) * vt * dx

    Ft = Ft_time \
        - AkVt * (u - ut) * vt * dx \
        + consumption * vt * dx \
        + difD * inner(grad(ut), grad(vt)) * dx \
        + Dmb * CMb * inner(grad(ut / (ut + C50)), grad(vt)) * dx

    CB = variable(4.0 * CHb * Hct * SHb(mesh, u_safe, pO2C))
    CT = CB + u_safe
    Fb = (ww[0] * weakL(v, u_safe, CT) + ww[1] * weakL(v, u_safe, CT)) * dx \
        - (ww[0] * AkVb * ut * v + ww[1] * AkVb * ut * v) * dx

    if steadySUPG == 1:
        Pe_temp = advU * dL / (2 * (difD + difDHb))
        tau = (dL / (2.0 * advU)) * (1.0 / tanh(Pe_temp) - 1.0 / Pe_temp) * W_inv
        tauPw = tau * Pw
        tauPwR = tauPw * funR(u, CT, ut)
        tauPwR = variable(tauPwR)  # TODO
        Fb += (tauPwR[0] + tauPwR[1]) * dx

    F = Fb + Ft
    J = derivative(F, uut, dudut)
    '''
    problem = NonlinearVariationalProblem(F, uut, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["relaxation_parameter"] = 0.02
    solver.parameters["newton_solver"]["report"] = True
    solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-13
    solver.parameters["newton_solver"]["maximum_iterations"] = 1000
    '''
    '''
    if timestep > 1:
        solver.parameters["newton_solver"]["relaxation_parameter"] = 0.1
    if timestep > 2:
        solver.parameters["newton_solver"]["relaxation_parameter"] = 0.05
    '''
    problem = NonlinearVariationalProblem(F, uut, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["nonlinear_solver"] = "snes"
    solver.parameters["snes_solver"]["report"] = True
    solver.parameters["snes_solver"]["absolute_tolerance"] = 1e-13
    solver.parameters["snes_solver"]["maximum_iterations"] = 1000
    solver.parameters["snes_solver"]["linear_solver"] = "lu"  # or "cg", "gmres", etc.
    solver.solve()

    # Print residual norm after SNES solve for direct comparison with PETSc SNES function norm
    res_postsolve = assemble(F)
    print(f"[Diagnostic] Residual norm after SNES solve: {res_postsolve.norm('l2'):.3e}")

    _, ut_new = uut.split(deepcopy=True)
    ut_old.assign(ut_new)

    # Optional: write to file
    xdmf_out = XDMFFile(commMPI, f"./results_1876v/CFb_step_{4+1:02d}.xdmf")
    u0, u1 = uut.split()
    u0.rename("CFb", "")
    xdmf_out.write(u0)
    xdmf_out.close()
    xdmf_out1 = XDMFFile(commMPI, f"./results_1876v/CFt_step_{4+1:02d}.xdmf")
    u1.rename("CFt", "")
    xdmf_out1.write(u1)
    xdmf_out1.close()


# Print elapsed time
print(f"Total runtime: {time() - start_time:.2f} s")
