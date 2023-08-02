A tutorial on writing custom restraints using the [PMI](https://github.com/salilab/pmi) interface of the [Integrative Modeling Platform](https://integrativemodeling.org/) software.

This was presented as a brief workshop at the [MASFE 2023](https://sites.google.com/acads.iiserpune.ac.in/masfe/) conference.

The custom restraints used here are taken from the [`nblib` library](https://github.com/integrativemodeling/nbspike/tree/main/nblib), written for modeling nanobody epitopes on target receptors (published [here](https://elifesciences.org/articles/73027)).

- ```restraints.py``` implments two custom restraints written using the ```RestraintBase``` PMI class. ```CenterPatchRestraint``` constrains the min. distance between a potential epitope center on a receptor with the paratope region of the binder molecule (in this case the nanobody). ```PatchPatchRestraint``` approximates shape complementarity between a proposed epitope on the target and the paratope on the nanobody. The epitope center on the receptor is essentially a escape mutation in the context of viral epitopes.

- ```modeling.py``` is a driver script that combines the above restraints with the usual ```ExcludedVolumeSphere``` and ```ConnectivityRestraint``` restraints in PMI to determine the binding mode of a given nanobody on the SARS-CoV-2 spike protein Receptor Binding Domain (RBD).

- ```data``` contains the topology, pdb and escape mutation data required for modeling. 

- ```demo_<date>_..``` folders contained short runs and ChimeraX sessions (called ```playback.cxs```) from sampling using a single starting conformation.

