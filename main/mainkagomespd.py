try:
    from spd.SparsePauliDynamics import *
    from spd.BaseOperatorRepresentation import * 
except ImportError:
    print("\n" + "="*60)
    print("REQUIRED PACKAGE 'spd' NOT FOUND")
    print("="*60)
    print("This notebook relies on the 'spd' package.")
    print("Please install it before proceeding:\n")
    print("   pip install git+https://github.com/tbegusic/spd.git")
    print("="*60 + "\n")
    raise

import numpy as np
import matplotlib.pyplot as plt
from qiskit_nature.second_q.hamiltonians.lattices import (
    KagomeLattice,
    BoundaryCondition,
)
from qiskit.quantum_info import SparsePauliOp


Lx, Ly = 3, 2
kagome = KagomeLattice(rows=Lx, cols=Ly,
    boundary_condition=(BoundaryCondition.PERIODIC, BoundaryCondition.PERIODIC)
)
nq = kagome.num_nodes
weighted_edges = kagome.weighted_edge_list
nlist = [(i, j) for i, j, _ in weighted_edges if i != j]

hc = 1
h = hc
hzz = SparsePauliOp.from_sparse_list([("ZZ", pair, 1.0) for pair in nlist], num_qubits=nq)
hx = SparsePauliOp.from_sparse_list([("X", [i], h) for i in range(nq)], num_qubits=nq)
h_tot = hzz + hx
obs = SparsePauliOp.from_sparse_list([("Z", [(nq-1)//2], 1.0)], num_qubits=nq)

totalt = 1
dt = 0.02
nsteps = int(totalt/dt)


def exp_val_func(obs):
    exp_val = np.sum(obs.coeffs[obs.ztype()])
    print(exp_val, obs.size)
    return exp_val

ops = dt*h_tot #trotterized evoltion operator
threshold = [2.**-14, 2.**-16, 2.**-18, 2.**-20]
results = {}
for thresh in threshold:
    sim = Simulation.from_pauli_list(obs, ops, threshold=thresh, nprocs=1)
    r = sim.run_dynamics(nsteps, process=exp_val_func, process_every = totalt)
    results[str(thresh)] = np.array(r)

np.savez('kagomespd_results.npz',**results)