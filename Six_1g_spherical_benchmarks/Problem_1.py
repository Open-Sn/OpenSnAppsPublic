#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 1
Chang, B. (2021a, February 19).
Six 1D polar transport test problems for the deterministic and monte-carlo method.
LLNL-TR-819680
https://www.osti.gov/servlets/purl/1769096
"""

import sys
import os
import numpy as np

if "opensn_console" not in globals():
    from mpi4py import MPI
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    # Append parent directory to locate the pyopensn modules
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import FromFileMeshGenerator, PETScGraphPartitioner
    from pyopensn.xs import MultiGroupXS
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSolver
    from pyopensn.fieldfunc import FieldFunctionInterpolationVolume
    from pyopensn.logvol import SphereLogicalVolume, BooleanLogicalVolume

if __name__ == "__main__":

    meshgen = FromFileMeshGenerator(
        filename="./meshes/single_sphere.msh",
        partitioner=PETScGraphPartitioner(type='parmetis'),
    )
    grid = meshgen.Execute()

    # Get measured volumes
    volumes_per_block = grid.ComputeVolumePerBlockID()

    # Define radii per block-ID
    radii = {1: 1.0}

    # Sort block IDs by increasing radius
    block_ids = sorted(volumes_per_block.keys())
    if rank == 0:
        print(block_ids)

    # Check for missing radii
    missing = [blk for blk in block_ids if blk not in radii]
    if missing:
        raise KeyError(f"No radius provided for block IDs: {missing}")

    # Compute exact volumes
    exact_volumes_per_block = {}
    prev_R = 0.0
    for blk in block_ids:
        R = radii[blk]
        exact_volumes_per_block[blk] = (4.0 / 3.0) * np.pi * (R ** 3 - prev_R ** 3)
        prev_R = R

    # Build the ratios array
    ratios = np.array([
        exact_volumes_per_block[blk] / volumes_per_block[blk]
        for blk in block_ids
    ])

    if rank == 0:
        for idx, blk in enumerate(block_ids):
            print(f"Block {blk}: measured = {volumes_per_block[blk]:.11e}, "
                  f"exact = {exact_volumes_per_block[blk]:.11e}, "
                  f"ratio = {ratios[idx]:.6f}")

    # Create and modify XS
    num_groups = 1
    xs_mat = MultiGroupXS()
    sigt = 1.0
    c = 0.0
    xs_mat.CreateSimpleOneGroup(sigma_t=sigt, c=c)

    # Boundary source
    bsrc = [1.0]

    # Angular quadrature
    nazimu = 4
    npolar = 2
    pquad = GLCProductQuadrature3DXYZ(npolar, nazimu)

    # Solver
    phys = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=[
            {
                "groups_from_to": (0, num_groups - 1),
                "angular_quadrature": pquad,
                "inner_linear_method": "petsc_gmres",
                "angle_aggregation_type": "single",
                "angle_aggregation_num_subsets": 1,
                "l_max_its": 500,
                "l_abs_tol": 1.0e-6,
                "gmres_restart_interval": 30,
            },
        ],
        xs_map=[
            {"block_ids": [1], "xs": xs_mat},
        ],
        options={
            "scattering_order": 0,
            "boundary_conditions": [
                {"name": "xmin", "type": "isotropic", "group_strength": bsrc},
                {"name": "xmax", "type": "isotropic", "group_strength": bsrc},
                {"name": "ymin", "type": "isotropic", "group_strength": bsrc},
                {"name": "ymax", "type": "isotropic", "group_strength": bsrc},
                {"name": "zmin", "type": "isotropic", "group_strength": bsrc},
                {"name": "zmax", "type": "isotropic", "group_strength": bsrc},
            ],
        },
    )

    ss_solver = SteadyStateSolver(lbs_problem=phys)
    ss_solver.Initialize()
    ss_solver.Execute()

    # Export
    fflist = phys.GetFieldFunctions()

    # Compute the average flux over a logical volume
    def average_flx_logvol(logvol):
        ffvol = FieldFunctionInterpolationVolume()
        ffvol.SetOperationType("avg")
        ffvol.SetLogicalVolume(logvol)
        ffvol.AddFieldFunction(fflist[0])
        ffvol.Initialize()
        ffvol.Execute()
        avgval = ffvol.GetValue()
        return avgval

    # Compute the average flux over spherical shells
    def compute_average_flux(N_logvols, r_max):
        r_vals = np.linspace(0, r_max, N_logvols + 1)
        avg_flx = np.zeros(N_logvols)
        for i in range(N_logvols):
            if i != 0:
                inner_vol = SphereLogicalVolume(r=r_vals[i])
                outer_vol = SphereLogicalVolume(r=r_vals[i + 1])
                logvol = BooleanLogicalVolume(parts=[
                    {"op": True, "lv": outer_vol},
                    {"op": False, "lv": inner_vol},
                ])
            else:
                logvol = SphereLogicalVolume(r=r_vals[i + 1])
            avg_flx[i] = average_flx_logvol(logvol)
        return r_vals, avg_flx

    # Perform the calculation
    N_logvols = 15
    r_vals, avg_flx = compute_average_flux(N_logvols, 1)
    if rank == 0:
        for r, v in zip(r_vals[1:], avg_flx):
            print("end-radius: {:.2f}, avg-value: {:.6f}".format(r, v))

    # Compute the analytical answer
    from scipy.special import exp1

    def E2(x):
        return np.exp(-x) - x * exp1(x)

    def get_phi(r, psi, r_max, sig):
        term1 = (np.exp(-sig * (r_max - r)) - np.exp(-sig * (r_max + r))) / sig
        term2 = (r + r_max) * E2(sig * (r_max - r))
        term3 = (r_max - r) * E2(sig * (r_max + r))
        phi = psi / (2 * r) * (term1 + term2 - term3)
        return phi

    if rank == 0:
        r_max = radii[1]
        eps = 1e-3
        psi = 2 * np.pi
        r_fine = np.linspace(0.0 + eps, r_max - eps, 100)
        exact_phi = np.zeros_like(r_fine)
        for i, r in enumerate(r_fine):
            exact_phi[i] = get_phi(r, psi, r_max, sigt)

        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(r_fine, exact_phi, label='Exact')
        plt.step(r_vals, np.append(avg_flx, avg_flx[-1]), where='post',
                 label='Radial Average Flux in Spherical Shells')
        plt.xlabel("Radius")
        plt.ylabel("Average Flux")
        plt.title("Flux as a function of radius")
        plt.grid(True)
        plt.legend()
        plt.savefig("Problem_1.png", dpi=300, bbox_inches='tight')
        plt.show()
