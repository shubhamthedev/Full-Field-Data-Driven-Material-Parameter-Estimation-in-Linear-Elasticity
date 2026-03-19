from dolfinx import fem, default_scalar_type
from pyamg import smoothed_aggregation_solver
from mpi4py import MPI
import ufl
import numpy as np
from dolfinx.io import gmshio
from dolfinx.fem import locate_dofs_topological
import pandas as pd
import os

def generate_snapshots(num_snapshots, G_range, K_range, save_folder):
    """
    Generate a specified number of snapshots for training.
    
    Parameters:
    - num_snapshots: Number of snapshots to generate.
    - G_range: Tuple indicating the range for shear modulus (min, max).
    - K_range: Tuple indicating the range for bulk modulus (min, max).
    - save_folder: Folder to save the snapshots.
    """
    # Create folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # Generate random combinations of G and K
    G_values = np.random.uniform(G_range[0], G_range[1], num_snapshots)
    K_values = np.random.uniform(K_range[0], K_range[1], num_snapshots)

    for i, (G, K) in enumerate(zip(G_values, K_values)):
        print(f"Generating snapshot {i+1}/{num_snapshots}: G={G:.2e}, K={K:.2e}")
        FEM_sol(G, K, save_folder=save_folder)

    print(f"All {num_snapshots} snapshots saved in {save_folder}")

def generate_test_snapshot(G,K,save_folder):
    # Create folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # Generate random combinations of G and K
    G_values = G
    K_values = K

    FEM_sol(G, K, save_folder=save_folder)

    print(f"Test snapshot saved in {save_folder}")

def FEM_sol(G, K, save_folder=None):
    """
    FEM Solver with snapshot saving for displacement fields.

    Parameters:
    - G: Shear modulus.
    - K: Bulk modulus.
    - save_folder: Folder to save snapshots (if None, no snapshots are saved).
    """
    # Define a function space
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

    # Define the Dirichlet Boundary Conditions
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    b_D = locate_dofs_topological(V, fdim, facet_tags.find(3))
    u_D = np.array([0, 0, 0], dtype=default_scalar_type)
    bc_D = fem.dirichletbc(u_D, b_D, V)

    # Define the Neumann Boundary Conditions
    T_neuman = fem.Constant(domain, default_scalar_type((106.26e6, 0, 0)))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    # Define the weak form of the problem
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u):
        return (-2 / 3 * G + K) * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * G * epsilon(u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, default_scalar_type((0, 0, 0)))
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(T_neuman, v) * ds(4)

    # Create and solve the linear problem using pyamg
    A = fem.assemble_matrix(a, [bc_D])
    b = fem.assemble_vector(L)
    fem.apply_lifting(b, [a], [[bc_D]])
    b.ghostUpdate()
    
    ml = smoothed_aggregation_solver(A)
    uh = ml.solve(b.array)

    # Post Processing
    x_origin_sub = [20e-3, 100e-3]
    y_origin_sub = 10e-3
    domain_arr = domain.geometry.x
    subdomain_condition = (domain_arr[:, 0] >= x_origin_sub[0]) & (domain_arr[:, 0] <= x_origin_sub[1])
    num_dofs = V.dofmap.index_map.size_local
    uh_arr = uh.reshape((num_dofs, -1))

    # Get the subdomain and displacements
    domain_sub = domain_arr[subdomain_condition]
    domain_sub[:, 0] -= x_origin_sub[0]
    domain_sub[:, 1] += y_origin_sub
    u_sub = uh_arr[subdomain_condition]

    # Save snapshots if a folder is specified
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        # Save combined displacement data in a single file
        filename = os.path.join(save_folder, f"snapshot_G{G:.2e}_K{K:.2e}.csv")
        # Combine x, y coordinates and both displacement components
        data = np.column_stack((
            domain_sub[:, 0],  # x-coordinate
            domain_sub[:, 1],  # y-coordinate
            u_sub[:, 0],      # x-displacement
            u_sub[:, 1]       # y-displacement
        ))
        # Save with headers matching the experimental data format
        header = "x-coordinate [m],y-coordinate [m],x-displacement [m],y-displacement [m]"
        np.savetxt(filename, data, delimiter=",", header=header, comments="")
        print(f"Saved snapshot: {filename}")

    return domain, uh, domain_sub, u_sub

if __name__ == "__main__":  # Fixed syntax
    domain, cell_tags, facet_tags = gmshio.read_from_msh('tensile_test_specimen.msh', MPI.COMM_WORLD, 0)

    # Create a folder for snapshots
    snapshot_folder = r"C:\Users\z004wesy\Desktop\surrogate_modelling\snapshots"
    test_folder = r"C:\Users\z004wesy\Desktop\surrogate_modelling\test_snapshot"

    generate_test_snapshot(
        G = 7.35e10, 
        K = 1.28e11,
        save_folder = test_folder
        )

    generate_snapshots(
        num_snapshots=50,
        G_range=(5e10, 1e11),    # 50-100 GPa
        K_range=(8e10, 1.5e11),  # 80-150 GPa
        save_folder=snapshot_folder
    )