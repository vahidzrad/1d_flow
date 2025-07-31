#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""
@author: Haifeng Wang (haifeng.wang@rub.de)
"""
import os
import tempfile
import zipfile
import shutil
import sys

import numpy as np
import math

from mpi4py import MPI
from petsc4py import PETSc

import ufl
from ufl.operators import sqrt
from ufl import (dot, grad, as_vector, variable, VectorElement, max_value)
from dolfin import *
#import h5py

#########################################################
def remove_recreate_folder(folder_name):
    current_dir = os.getcwd()
    folder_path = os.path.join(current_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")
        shutil.rmtree(folder_path)  #To avoid HD5-issue!
        print(f"Folder '{folder_name}' removed.")
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' recreated.")
#########################################################
def create_folder_if_not_exists(folder_name):
    current_dir = os.getcwd()
    folder_path = os.path.join(current_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created.")

#########################################################
# Function to save simulation state
def save_state(u, filename="./stateBackup/simulation_state.npz"):
    # Convert to numpy arrays
    u_array = u.vector[:]
    # Save using np.savez
    np.savez(filename, u=u_array)

def load_state(u, filename="./stateBackup/simulation_state.npz"):
    # Load the data
    with np.load(filename, allow_pickle=False) as data:
        #step = data['step']
        u.vector.setArray(data['u'])
        #un.vector.setArray(data['un'])
    #return step

# MPI-VERSION:
def save_state_mpi(comm, step, u, filename_template="./stateBackup/simulation_state_rank{}.npz"):
    rank = comm.Get_rank()
    
    # Each rank saves its own local portion of u and un
    u_local_array = u.vector.array

    # Generate filename specific to each rank
    filename = filename_template.format(rank)
    
    # Save each rank's local data separately
    np.savez(filename, step=np.array([step]), u=u_local_array)

    #print(f"Rank {rank}: State saved to {filename}")

def load_state_mpi(comm, filename_template="./stateBackup/simulation_state_rank{}.npz", u=None):
    rank = comm.Get_rank()
    
    # Generate rank-specific filename
    filename = filename_template.format(rank)

    try:
        # Load each rank’s corresponding data
        data = np.load(filename, allow_pickle=False)
        step = int(data["step"][0])  # Extract step (only one element)
        u.vector.array[:] = data["u"]
        #print(f"Rank {rank}: Loaded state from {filename}")
    except FileNotFoundError:
        print(f"Rank {rank}: No saved state found. Starting fresh.")
        step = 0

    # Ensure all processes have the same step number
    step = comm.bcast(step, root=0)





def save_state_hdf5(step, u, un, filename="./stateBackup/simulation_state.h5"):
    """   
    - Each MPI process writes its own function data.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    with h5py.File(filename, "w", driver="mpio", comm=comm) as f:
        # Rank 0 stores the current simulation step as an attribute
        if rank == 0:
            f.attrs["step"] = step
        
        # Each MPI rank saves its local part of u and un
        f.create_dataset(f"u_{rank}", data=u.x.array)
        f.create_dataset(f"un_{rank}", data=un.x.array)

    comm.Barrier()  # Ensure all ranks complete saving

def load_state_hdf5(u, un, filename="./stateBackup/simulation_state.h5"):
    """    
    - Each MPI process loads its own part of u and un.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    with h5py.File(filename, "r", driver="mpio", comm=comm) as f:
        # Rank 0 reads the step number and broadcasts it to all ranks
        step = f.attrs["step"] if rank == 0 else None
        step = comm.bcast(step, root=0)

        # Each rank loads its respective portion of u and un
        u.x.array[:] = f[f"u_{rank}"][:]
        un.x.array[:] = f[f"un_{rank}"][:]

    comm.Barrier()  # Ensure all ranks finish loading before continuing
    return step


#########################################################
tolCoord = 1e-5  # Tolerance for coordinate comparison
#########################################################
class GaussianHill:
    def __init__(self):
        self.t = 0.0
    def eval(self, x):
        x0tmp = 0.3
        Ltmp = x0tmp / 10.0  
        Btmp = 1.35e-12 * 95.0  #[mol/mm^3]
        return np.full(x.shape[1], Btmp * np.exp(-((x[1] - x0tmp) / Ltmp) ** 2))

##################################################################
class DeltaPulse:
    def __init__(self):
        self.t = 0.0  # Time attribute which might be used later
    def eval(self, x):
        x0tmp = 0.5
        rlTmp = 5
        pO2C = 1.35E-12  #pO2[mmHg] = C[mol/mm^3] / alpha[(mol/mm^3)/mmHg]; C=pO2*alpha
        coefTmp = 50 * pO2C /rlTmp  #pO2=C/alpha; C=pO2*alpha
        std_dev = 0.01 /rlTmp
        return np.full(x.shape[1], coefTmp * np.exp(-((x[1] - x0tmp) ** 2) / (2 * std_dev ** 2)))   
        
    
#########################################################
def assign_local_property_vertexBased(mesh, value, V):
    kW = Function(V)
    dof_coords = V.tabulate_dof_coordinates()
    dof_coords = dof_coords.reshape((-1, mesh.geometry().dim()))

    # Create array of values
    values = np.full(len(dof_coords), value)

    for i, coord in enumerate(dof_coords):
        if coord[2] > 0:  # z > 0
            values[i] = 0.0

    # Assign values to the Function
    kW.vector()[:] = values

    return kW

def assign_local_property_vertexBased_celltags(mesh, value, V, cell_tags):
    kW = Function(V)
    dof_coords = V.tabulate_dof_coordinates()
    dof_coords = dof_coords.reshape((-1, mesh.geometry().dim()))

    # Create array of default values
    values = np.full(len(dof_coords), value)

    # Map each dof to a cell and inspect cell tag
    dofmap = V.dofmap()
    for cell in cells(mesh):
        tag = cell_tags[cell.index()]
        if tag > 585:
            # Get dofs for this cell
            cell_dofs = dofmap.cell_dofs(cell.index())
            for dof in cell_dofs:
                values[dof] = 0.0  # Overwrite value

    # Assign values to Function
    kW.vector()[:] = values

    return kW

def assign_local_property_vertexBased_uin(domain, value, V, cell_tags):

# Output function
    kW = Function(V)

    # Build vertex-to-cell connectivity
    mesh.init(0, mesh.topology().dim())  # vertex-to-cell
    v2c = mesh.topology()(0, mesh.topology().dim())

    # Get map from vertex index to DOF index
    dofmap = V.dofmap()
    vertex_to_dof = vertex_to_dof_map(V)

    # Initialize all values to 0.5 * value
    values =  value * np.ones(V.dim())

    for v in range(mesh.num_vertices()):
        if v2c.size(v) == 0:
            continue
        connected_cells = v2c(v)
        for cell_idx in connected_cells:
            vessel_id = cell_tags[cell_idx]
            if vessel_id > 585:
                dof = vertex_to_dof[v]
                values[dof] = 0 * value
                break  # early exit

    # Assign to function
    kW.vector()[:] = values
    kW.vector().apply("insert")

    return kW

def assign_initial_condition_vertex_based(mesh, V, value, y_threshold=0.3):
    """
    Assigns initial conditions to a CG1 Function based on vertex coordinates.
    :param mesh: The mesh
    :param V: FunctionSpace (CG1)
    :param value: Base value (float)
    :param y_threshold: y-coordinate threshold to apply different values
    :return: Function with assigned values
    """
    init_func = Function(V)
    dof_coords = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
    dof_to_vertex = V.dofmap().dofs()

    local_values = init_func.vector().get_local()

    for i, coord in enumerate(dof_coords):
        if coord[2] > 0:  # z > 0
            if coord[1] < y_threshold:
                local_values[i] = 0.90 * value
            else:
                local_values[i] = 0.20 * value
        else:
            local_values[i] = 0.6 * value

    init_func.vector().set_local(local_values)
    init_func.vector().apply("insert")

    return init_func

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

def assign_local_property_vertexBased_uin_scaled(domain, value, V):
    kW = fem.Function(V)
    num_vertices = domain.topology.index_map(0).size_local
    vertex_coords = domain.geometry.x

    # Initialize an array with default values
    kW_values = np.full(num_vertices, 0.5 * value, dtype=PETSc.ScalarType)

    # Define scaling parameters
    z_min_1, z_max_1 = 0.00001, 153.330  # First case range
    z_min_2, z_max_2 = 0.00001, 194.159  # Second case range
    
    v_min_1, v_max_1 = 0.8 * value, 0.9 * value  # First case: scale from 0.9*value to 0.5*value
    v_min_2, v_max_2 = 0.2 * value, 0.3 * value  # Second case: scale from 0.3*value to 0.2*value

    def linear_scale(z, z_min, z_max, v_min, v_max):
        """Performs linear interpolation between min/max values based on z-coordinate."""
        return v_min + ((z - z_min) / (z_max - z_min)) * (v_max - v_min) if z_min < z < z_max else v_min if z <= z_min else v_max

    # Iterate over all vertex coordinates
    for vertexID in range(num_vertices):
        z = vertex_coords[vertexID, 2]

        if z > 0:
            if vertex_coords[vertexID, 1] < 0.3:
                kW_values[vertexID] = linear_scale(z, z_min_1, z_max_1, v_min_1, v_max_1)
            elif vertex_coords[vertexID, 1] > 0.3:
                kW_values[vertexID] = linear_scale(z, z_min_2, z_max_2, v_max_2, v_min_2)

    kW.vector.setArray(kW_values)

    # Update ghost values for parallel computing
    if MPI.COMM_WORLD.Get_size() > 1:
        kW.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    return kW


def assign_local_property_vertexBased_Gmax(domain, value, V):
    kW = fem.Function(V)
    
    # Get the total number of vertices in the mesh
    num_vertices = domain.topology.index_map(0).size_local + domain.topology.index_map(0).num_ghosts
    vertex_coords = domain.geometry.x
    
    # Initialize an array to hold the new values
    kW_values = np.full(num_vertices, value, dtype=PETSc.ScalarType)
    
    # Iterate over all vertex coordinates
    for vertexID in range(num_vertices):
        # Check the z-coordinate of each vertex
        if vertex_coords[vertexID, 2] > 0:
            kW_values[vertexID] = 0.0
        
        #if vertex_coords[vertexID, 1] < 1:
        #    if vertex_coords[vertexID, 1] > -0.1:
        #        kW_values[vertexID]=kW_values[vertexID]*0.5
        
    
    # Get the dofmap to relate vertices to DoFs for CG-1 space
    dofmap = V.dofmap.list.array
    
    # Directly assign the modified values to the function's vector
    with kW.vector.localForm() as loc:
        loc.setValues(dofmap, kW_values[dofmap])

    # Update ghost values
    kW.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    return kW

def assign_local_property_vertexBased_difD(domain, value, V):
    difD = fem.Function(V)
    
    # Get the total number of vertices in the mesh
    num_vertices = domain.topology.index_map(0).size_local + domain.topology.index_map(0).num_ghosts
    vertex_coords = domain.geometry.x
    
    # Initialize an array to hold the new values
    kW_values = np.full(num_vertices, value, dtype=PETSc.ScalarType)
    
    mask = (
        (0.355595 <= vertex_coords[:, 0]) & (vertex_coords[:, 0] <= 1.105595) &
        (47.49861 <= vertex_coords[:, 2]) & (vertex_coords[:, 2] <= 100.3986)
    )
    kW_values[mask] *= 0.01
    
    # Get the dofmap to relate vertices to DoFs for CG-1 space
    dofmap = V.dofmap.list.array
    
    # Directly assign the modified values to the function's vector
    with difD.vector.localForm() as loc:
        loc.setValues(dofmap, kW_values[dofmap])

    # Update ghost values
    difD.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    return difD
#########################################################
#########################################################
def find_vertices_directly_connected_to_vertexi(vertex_index, mesh):
    tdim = mesh.topology.dim  # Topological dimension of the mesh
    mesh.topology.create_connectivity(0, tdim)  # Ensure vertex-to-cell connectivity is created
    mesh.topology.create_connectivity(tdim, 0)  # Ensure cell-to-vertex connectivity is created
    v_to_c = mesh.topology.connectivity(0, tdim)  # Vertex-to-cell connectivity
    c_to_v = mesh.topology.connectivity(tdim, 0)  # Cell-to-vertex connectivity

    connected_vertices = set()
    for cell in v_to_c.links(vertex_index):  # Cells connected to the vertex
        for v in c_to_v.links(cell):  # Vertices connected to the cell
            if v != vertex_index:
                connected_vertices.add(v)
    return list(connected_vertices)

#########################################################

def create_solver(target_space, petsc_options=None):
    u = TrialFunction(target_space)
    v = TestFunction(target_space)
    a = inner(u, v) * dx

    A = assemble(a)

    # Create solver
    solver = PETScKrylovSolver()
    solver.set_operator(A)

    if petsc_options:
        for k, v in petsc_options.items():
            PETScOptions.set(k, v)

    return solver, A



def project_function(target_space, source_function, solver, A):
    # Define the linear form for projection
    v = ufl.TestFunction(target_space)
    L = ufl.inner(source_function, v) * ufl.Measure('dx', domain=target_space.mesh)

    # Wrap the linear form with dolfinx.fem.Form
    L_form = fem.form(L)

    # Assemble the RHS vector
    b = fem.petsc.assemble_vector(L_form)
    #fem.petsc.apply_lifting(b, L_form, bcs=[])
    # Update ghost values
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    A.assemble()
    # Create a function to store the solution
    target_function = fem.Function(target_space)

    # Solve the linear system
    solver.solve(b, target_function.vector)
    target_function.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    return target_function

##################
def cellDirVec_DG(mesh, dirVect_cells):
    DG_space = VectorFunctionSpace(mesh, "DG", 0)

    # Create a Function to hold the values
    vec_function = Function(DG_space)

    # Create a user-defined Expression that evaluates to the right vector per cell
    # NOTE: We use `cell.index()` to get the index of the current cell
    class DirVectorExpression(UserExpression):
        def __init__(self, dir_vectors, **kwargs):
            super().__init__(**kwargs)
            self.dir_vectors = dir_vectors

        def eval_cell(self, values, x, cell):
            v = self.dir_vectors[cell.index]
            values[0], values[1], values[2] = v

        def value_shape(self):
            return (3,)

    # Instantiate the expression
    expr = DirVectorExpression(dirVect_cells, degree=0)

    # Project into the DG space
    vec_function = project(expr, DG_space)

    return vec_function
#########################################################
from dolfin import *
import numpy as np
from mpi4py import MPI

def compute_directional_vectors_cells(mesh):
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size

    # Build connectivity from edges (dim=1) to vertices (dim=0)
    mesh.init(1, 0)
    edge_to_vertex = mesh.topology()(1, 0)

    num_edges_local = mesh.num_entities(1)
    directional_vectors = np.zeros((num_edges_local, 3))

    coordinates = mesh.coordinates()

    for edge_index in range(num_edges_local):
        vertex_indices = edge_to_vertex(edge_index)
        vertex_coords = coordinates[vertex_indices]

        if vertex_coords.shape[0] != 2:
            raise RuntimeError(f"Edge {edge_index} does not have 2 vertices")

        # Compute vector
        vector = vertex_coords[1, :] - vertex_coords[0, :]
        norm = np.linalg.norm(vector)

        if norm > 1e-12:
            vector /= norm
            directional_vectors[edge_index, :] = vector
        else:
            directional_vectors[edge_index, :] = np.array([0, 0, 0])
            print(f"--> norm = 0 for edge {edge_index}")
            exit(0)

    return directional_vectors

#########################################################
def compute_length(inlet_coords, outlet_coords):
    inlet_coords = np.squeeze(np.asarray(inlet_coords))
    outlet_coords = np.squeeze(np.asarray(outlet_coords))
    # Calculate the direction vector
    direction_vector = outlet_coords - inlet_coords
    # Calculate the length of the direction vector
    return np.sqrt(np.dot(direction_vector, direction_vector))

#########################################################
def SHb_DB2001(domain, CF, solCoef):
    solCoef = Constant(solCoef) 
    
    a1 = Constant( 0.01524)
    a2 = Constant( 2.7e-6)
    a3 = Constant( 0.0)
    a4 = Constant( 2.7e-6)
    
    alphaCF = CF / solCoef
    
    one = Constant( 1.0)
    two = Constant( 2.0)
    three = Constant( 3.0)
    four = Constant( 4.0)
    
    return (a1 * alphaCF + two * a2 * alphaCF ** two + three * a3 * alphaCF ** three + four * a4 * alphaCF ** four) / \
           (four * (one + a1 * alphaCF + a2 * alphaCF ** two + a3 * alphaCF ** three + a4 * alphaCF ** four))

#########################################################
'''
def SHb(domain, CF, solCoef):
    a, b, c, n = 0.34332, 0.64073, 0.34128, 1.58678  # Fitted over range 0<=SHb<=1
    solCoef = Constant(solCoef)  # [(mol/mm^3) /mmHg], keeps pO2 in mmHg!!!

    P50 = 26.8  # [mmHg] Half-saturation pressure of hemoglobin [ref: Dash2006]
    Temp= 37.0  # Temperature in degrees Celsius
    pH = 7.4    # pH value
    Pco2 = 40.0  # [mmHg] Partial pressure of CO2 in mmHg
    T_pH_Pco2 = 10 ** (0.024 * (37.0 - Temp) + 0.4 * (pH - 7.4) + 0.06 * np.log(40.0 / Pco2))
    raw_pO2 = CF / solCoef
    #pO2 = conditional(gt(raw_pO2, 0.0), raw_pO2, 1e-10)
    #pO2 = conditional(gt(raw_pO2, 1e-16), raw_pO2, 1e-16)  # prevent near-zero pO2

    pO2 = CF / solCoef  # [mmHg]
    xTmp = T_pH_Pco2 * pO2 / P50
    SHb = (a * xTmp ** n + b * xTmp ** (2 * n)) / (1 + c * xTmp ** n + b * xTmp ** (2 * n))
    return SHb

'''
def SHb(domain, CF, solCoef):
    P50 = 26.8  # mmHg
    n = 2.7     # Hill coefficient (typical physiological range: 2–3)

    # Regularized pO2 value
    pO2 = CF / Constant(solCoef)

    # Classic Hill saturation curve
    SHb = (pO2**n) / (P50**n + pO2**n)

    # Optional: force bounds to [0, 1]
    #SHb = ufl.min_value(1.0, ufl.max_value(0.0, SHb))
    return SHb


class NonlinearPDE_SNESProblem():
    def __init__(self, F, J, soln_vars, bcs, P=None):
        self.L = F
        self.a = J
        self.a_precon = P
        self.bcs = bcs
        self.soln_vars = soln_vars

    def F_mono(self, snes, x, F):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with x.localForm() as _x:
            self.soln_vars.x.array[:] = _x.array_r
        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, self.L)
        apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, self.bcs, x, -1.0)

    def J_mono(self, snes, x, J, P):
        J.zeroEntries()
        assemble_matrix(J, self.a, bcs=self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            assemble_matrix(P, self.a_precon, bcs=self.bcs, diagonal=1.0)
            P.assemble()

    def F_block(self, snes, x, F):
        assert x.getType() != "nest"
        assert F.getType() != "nest"
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with F.localForm() as f_local:
            f_local.set(0.0)

        offset = 0
        x_array = x.getArray(readonly=True)
        for var in self.soln_vars:
            size_local = var.vector.getLocalSize()
            var.vector.array[:] = x_array[offset: offset + size_local]
            var.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            offset += size_local

        assemble_vector_block(F, self.L, self.a, bcs=self.bcs, x0=x, scale=-1.0)

    def J_block(self, snes, x, J, P):
        assert x.getType() != "nest" and J.getType() != "nest" and P.getType() != "nest"
        J.zeroEntries()
        assemble_matrix_block(J, self.a, bcs=self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            assemble_matrix_block(P, self.a_precon, bcs=self.bcs, diagonal=1.0)
            P.assemble()

    def F_nest(self, snes, x, F):
        assert x.getType() == "nest" and F.getType() == "nest"
        # Update solution
        x = x.getNestSubVecs()
        for x_sub, var_sub in zip(x, self.soln_vars):
            x_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            with x_sub.localForm() as _x:
                var_sub.x.array[:] = _x.array_r

        # Assemble
        bcs1 = bcs_by_block(extract_function_spaces(self.a, 1), self.bcs)
        for L, F_sub, a in zip(self.L, F.getNestSubVecs(), self.a):
            with F_sub.localForm() as F_sub_local:
                F_sub_local.set(0.0)
            assemble_vector(F_sub, L)
            apply_lifting(F_sub, a, bcs=bcs1, x0=x, scale=-1.0)
            F_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Set bc value in RHS
        bcs0 = bcs_by_block(extract_function_spaces(self.L), self.bcs)
        for F_sub, bc, x_sub in zip(F.getNestSubVecs(), bcs0, x):
            set_bc(F_sub, bc, x_sub, -1.0)

        # Must assemble F here in the case of nest matrices
        F.assemble()

    def J_nest(self, snes, x, J, P):
        assert J.getType() == "nest" and P.getType() == "nest"
        J.zeroEntries()
        assemble_matrix_nest(J, self.a, bcs=self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            assemble_matrix_nest(P, self.a_precon, bcs=self.bcs, diagonal=1.0)
            P.assemble()

#########################################################
if __name__ == "__main__":
    print('--> auxiliaryFunctions!')
#########################################################