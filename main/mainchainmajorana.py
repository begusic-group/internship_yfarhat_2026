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
from qiskit_nature.second_q.operators import FermionicOp


L = 100                   
t = 1.0
U = 1.0
dt = 0.12                   
T = 6.0                     
nsteps = int(T / dt)         

threshold = 1e-5


nmodes = 2 * L

nlist = [(i, i+1) for i in range(L-1)]


ham_hop = FermionicOp({}, nmodes)
ham_rep = FermionicOp({}, nmodes)


for i, j in nlist:
    up_i = 2 * i
    up_j = 2 * j
    ham_hop += FermionicOp({f'+_{up_i} -_{up_j}': -t}, num_spin_orbitals=nmodes)
    ham_hop += FermionicOp({f'+_{up_j} -_{up_i}': -t}, num_spin_orbitals=nmodes)
    down_i = 2 * i + 1
    down_j = 2 * j + 1
    ham_hop += FermionicOp({f'+_{down_i} -_{down_j}': -t}, num_spin_orbitals=nmodes)
    ham_hop += FermionicOp({f'+_{down_j} -_{down_i}': -t}, num_spin_orbitals=nmodes)

for site in range(L):
    up_idx = 2 * site
    down_idx = 2 * site + 1
    n_up = FermionicOp({f'+_{up_idx} -_{up_idx}': 1}, num_spin_orbitals=nmodes)
    n_down = FermionicOp({f'+_{down_idx} -_{down_idx}': 1}, num_spin_orbitals=nmodes)
    ham_rep += U * n_up @ n_down

central = 50              
obs_central_idx = 2 * central
obs_ferm = FermionicOp({f'+_{obs_central_idx} -_{obs_central_idx}': 1}, num_spin_orbitals=nmodes)
obs_majorana = MajoranaRepresentation.from_fermionic_op(obs_ferm)

h_hop_majorana = MajoranaRepresentation.fermionic_to_sparse_pauli_op(ham_hop)
h_rep_majorana = MajoranaRepresentation.fermionic_to_sparse_pauli_op(ham_rep)

op = (dt * h_hop_majorana) + (dt * h_rep_majorana)

state = np.zeros(nmodes, dtype=int)
for site in range(L):
    if site % 2 == 0:
        state[2*site+1] = 1 #even gets spin down
    else:
        state[2*site] = 1  #odd gets spin up

S_vals = [2, 4, 6, 8]
results = {}
string_counts = {}
for S in S_vals:
    countstr = []
    def process(obs_rep):
        """Compute expectation value of the observable representation with the initial state."""

        exp_val = MajoranaRepresentation.exp_val_comp_basis_state(obs_rep, state)

        countstr.append(obs_rep.bits.shape[0])

        trunc = obs_rep.nonp_count <= S
        obs_rep.bits = obs_rep.bits[trunc]
        obs_rep.coeffs = obs_rep.coeffs[trunc]

        return exp_val
    

    sim = Simulation(obs_majorana, op, threshold=threshold)

    r = sim.run_dynamics(nsteps, process=process, process_every=1)

    results[str(S)] = np.array(r)
    string_counts[str(S)] = np.array(countstr)


np.savez('chainmajorana_results.npy', **results)  
np.savez('chainmajorana_string_counts.npy', **string_counts)