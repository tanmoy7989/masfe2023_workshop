"""
Integrative modeling of the binary docking of a nanobody  
to the SARS-CoV2 Spike RBD, using escape mutation restraints.

Tanmoy Sanyal, PhD
Postdoctoral scholar @ Sali lab, UCSF, 2019-2022
"""

import os
import pandas as pd

import IMP
import IMP.atom
import IMP.pmi.topology
import IMP.pmi.restraints.stereochemistry
import IMP.pmi.macros
import IMP.pmi.tools

from restraints import PatchPatchRestraint, CenterPatchRestraint

import numpy as np
np.random.seed = int(10000 * np.random.random())

# data files
escape_mutation_csv_file = "data/escape_mutations.csv"

# topology file
topology_file = "data/topology.txt"

# output path
output_path = "output"
if not os.path.isdir(output_path):
    os.makedirs(output_path, exist_ok=True)

# restraint weights
evr_weight = 10.0
cr_weight = 1.0
ccr_weight = 1.0
ppr_weight = 0.001

# center-center restraint params
ccr_cutoff = 8.0 
ccr_kappa = 1.0

# patch-patch restraint params
ppr_epitope_cutoff = 15.0
ppr_cutoff = 10.0
ppr_kappa = 1.0


# --------------
# REPRESENTATION
# --------------
t = IMP.pmi.topology.TopologyReader(topology_file, pdb_dir="data/pdb", 
                                    fasta_dir="data")

components = t.get_components()
receptor, ligand = components[0].molname, components[1].molname

m = IMP.Model()
bs = IMP.pmi.macros.BuildSystem(m, resolutions=[1])
bs.add_state(t)
root_hier, dof = bs.execute_macro(
    max_rb_trans=1.0,
    max_rb_rot=0.05,
    max_bead_trans=2.00,
    max_srb_trans=1.00,
    max_srb_rot=0.05
)


restraint_list = []

# -------------------------
# EXCLUDED VOLUME RESTRAINT
# -------------------------
evr = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(
    included_objects=bs.get_molecule(receptor),
    other_objects=bs.get_molecule(ligand),
    resolution=1
)
evr.set_weight(evr_weight)
restraint_list.append(evr)


# ----------------------------------------------------
# CONNECTIVITY RESTRAINT
# (to handle missing residues in the receptor, if any)
# ----------------------------------------------------
cr = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(
    bs.get_molecule(receptor)
)
cr.set_weight(cr_weight)
restraint_list.append(cr)


# ----------------------------------
# ESCAPE MUATION DISTANCE RESTRAINTS
# ----------------------------------
# parse escape mutation data
df = pd.read_csv(escape_mutation_csv_file)
escape_mutation_data = []
for i in range(len(df)):
    e = int(df.iloc[i]["epitope_center"])
    
    p = []
    for (start, stop) in eval(df.iloc[i]["paratope"]):
        this_p = list(range(start, stop + 1))
        p.extend(this_p)
    escape_mutation_data.append((e, p))
    
# individual center-center restraints for each epitope center residue
for (e, p) in escape_mutation_data:
    ccr = CenterPatchRestraint(
        root_hier=root_hier, resolution=1,
        receptor_name=receptor, ligand_name=ligand,
        epitope_center_residue=e, 
        paratope_residues=p,
        kappa=ccr_kappa,
        cutoff=ccr_cutoff,
        label=str(e)
    )
    
    ccr.set_weight(ccr_weight)
    restraint_list.append(ccr)


# single patch-patch restraint for the entire epitope
epitope_center_residues, paratope_residues = [], []
for (e, p) in escape_mutation_data:
    epitope_center_residues.append(e)
    paratope_residues.extend(p)

paratope_residues = sorted(list(set(paratope_residues)))

ppr = PatchPatchRestraint(
    root_hier=root_hier, resolution=1,
    receptor_name=receptor, ligand_name=ligand,
    epitope_center_residues=epitope_center_residues,
    paratope_residues=paratope_residues,
    epitope_cutoff=ppr_epitope_cutoff,
    kappa=ppr_kappa,
    cutoff=ppr_cutoff,
    label="_".join([str(e) for e in epitope_center_residues])
)

ppr.set_weight(ppr_weight)
restraint_list.append(ppr)


# --------
# SAMPLING
# --------
# remove the rigid part of receptor from the degrees of freedom, 
# i.e. this will be unaffected by monte-carlo movers.
dof.disable_movers(objects=bs.get_molecule(receptor).get_atomic_residues())

# shuffle only ligand particles to randomize the system 
# i.e. don't re-initialize the receptor position between independent runs
IMP.pmi.tools.shuffle_configuration(
    bs.get_molecule(ligand),
    max_translation=500.0,
    niterations=100
)

# add all restraints to the model
for r in restraint_list:
    r.add_to_model()

# run replica exchange Monte-Carlo
rex = IMP.pmi.macros.ReplicaExchange0(
    m, root_hier,
    monte_carlo_sample_objects=dof.get_movers(),
    global_output_directory=output_path,
    output_objects=restraint_list,
    write_initial_rmf=True,

    monte_carlo_steps=100,
    number_of_frames=1000,
    number_of_best_scoring_models=0,

    simulated_annealing=True,
    simulated_annealing_minimum_temperature=1.0,
    simulated_annealing_maximum_temperature=2.5,

    monte_carlo_temperature=1.0,
    replica_exchange_minimum_temperature=1.0,
    replica_exchange_maximum_temperature=2.5
)
rex.execute_macro()
