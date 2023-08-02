"""
Patch-Patch and Center-Patch restraints for escape mutation data 
in modeling nanobody epitopes on pathogenic receptor targets.

Tanmoy Sanyal, PhD
Postdoctoral scholar @ Sali lab, UCSF, 2019-2022
"""

import IMP
import IMP.algebra
import IMP.atom
import IMP.container
import IMP.core
import IMP.pmi.tools
from IMP.pmi.restraints import RestraintBase

# slack for IMP close pair containers
# see: https://integrativemodeling.org/2.14.0/doc/ref/classIMP_1_1container_1_1CloseBipartitePairContainer.html#afa0bdd7250c318333de927ea873b11c6
_SLACK = 10.0

# Connolly surface params
_CONNOLLY_PROBE_RADIUS = 5.0 # A
_CONNOLLY_SURFACE_THICKNESS = 4.0 # A
_CONNOLLY_SAMPLING_DENSITY = 0.15


class PatchPatchRestraint(RestraintBase):
    def __init__(
        self, root_hier, resolution=1,
        
        receptor_name="receptor",
        ligand_name="ligand",
        
        epitope_center_residues=[],
        paratope_residues = [],
        epitope_cutoff=4.0,
        
        kappa=1.0, cutoff=4.0,
        
        weight=1.0, label=None,
        render=True
    ):

        """
        Restraint between epitope surface patch on receptor 
        and paratope surface patch on ligand. 
        
        Args:
        root_hier (IMP.hierarchy): Root hierarchy that provides access to all
        particles in the system.
        
        resolution (float, optional): Coarse grained resolution. Defaults to 1
        residue per bead.
        
        receptor_name (str, optional): Name of receptor molecule in topology.
        Defaults to "receptor".
        
        ligand_name (str, optional): Name of ligand molecule in topology.
        Defaults to "ligand".
        
        epitope_center_residues (list): Target residues on the receptor that
        roughly denote the centers of (multiple if applicable) epitopes. 
        Defaults to empty list, in which case the entire receptor surface
        is used. 
        
        paratope residues (list, optional): List of residues that form the 
        paratope on the ligand. Defaults to empty list, in which case
        the entire ligand is used as a valid paratope.
        
        epitope_cutoff (float, optional): Max. distance between epitope center
        residue and connolly surface points on the receptor, to include within
        epitope. These Connolly points will be then used to trace back which
        residues along the receptor surface are closest to the epitope center.
        
        kappa (float, optional): Strength of this restraint. Defaults to 1.0.
        
        cutoff (float, optional): Max. allowed value of the closest
        approach distance between epitope and paratope particles.
        Defaults to 4.0 A.
        
        weight (float, optional): Relative weight of this restraint relative to
        other restraints in the system. Defaults to 1.0.
        
        label (str, optional): Label for this restraint. Defaults to None.
        
        render (bool, optional): If true, this will write a BILD script
        that can be imported into ChimeraX to display the epitope and paratope
        patches.
        """
        
        # attributes that will be required throughout
        self.root_hier = root_hier
        self.resolution = resolution
        
        self.receptor_name = receptor_name
        self.ligand_name = ligand_name
        
        self.epitope_center_residues = epitope_center_residues
        self.paratope_residues = paratope_residues
        self.epitope_cutoff = epitope_cutoff
        
        self.render = render
        
        # get epitope patch
        ps_epitope, ps_buried = self.get_epitope()
        
        # get paratope patch
        ps_paratope = self.get_paratope()
        
        # init parent class
        self.model = ps_epitope[0].get_model()
        super().__init__(
            self.model,
            name=f"PatchPatchScore_{label}",
            label=label,
            weight=weight
        )
        
        # close bipartite container between epitope and paratope particles
        lsr = IMP.container.ListSingletonContainer(self.model)
        lsr.add(ps_epitope)
        
        lsl = IMP.container.ListSingletonContainer(self.model)
        lsl.add(ps_paratope)
        
        cpc = IMP.container.CloseBipartitePairContainer(
            lsr, lsl, 20.0, _SLACK
        )
        
        # pair score
        dps = IMP.core.HarmonicUpperBoundSphereDistancePairScore(cutoff, kappa)
        
        # restraint
        restraint = IMP.container.PairsRestraint(dps, cpc)
        self.rs.add_restraint(restraint)
        
        self._include_in_rmf = True
    
    
    def get_epitope(self):
        # 1. get receptor particles 
        sel = IMP.atom.Selection(
            hierarchy=self.root_hier, 
            resolution=self.resolution,
            molecule=self.receptor_name
        )
        ps = sel.get_selected_particles()
        
        # 2. get particle indices corresponding to given epitope center residue
        # if nothing is given, use entire receptor
        query_indices = []
        if self.epitope_center_residues:
            sel.set_residue_indexes(self.epitope_center_residues)
            query_ps = sel.get_selected_particles()
            query_indices = [ps.index(p) for p in query_ps]
        
        # 3. extract receptor surface particles, 
        # ie. particles on the Conolly surface of the receptor
        spheres, centers = [], []
        for p in ps:
            p_xyzr = IMP.core.XYZR(p)
            c, r = p_xyzr.get_coordinates(), p_xyzr.get_radius()
            centers.append(c)
            spheres.append(IMP.algebra.Sphere3D(c,r))

        csplist = IMP.algebra.get_connolly_surface(
            spheres, _CONNOLLY_SAMPLING_DENSITY, _CONNOLLY_PROBE_RADIUS
        )
        connolly_points = [p.get_surface_point() for p in csplist]
        
        # 4. filter the connolly points to ones 
        # that are within query radius of query indices
        if query_indices:
            nn = IMP.algebra.NearestNeighbor3D(connolly_points)
            connolly_indices = set()
            for i in query_indices:
                q = centers[i]
                this_ci = nn.get_in_ball(q, self.epitope_cutoff)
                connolly_indices |= set(this_ci)
            connolly_points = [connolly_points[i] for i in connolly_indices]

        # 5. partition receptor into surface and core particles
        surface_indices = []
        nn = IMP.algebra.NearestNeighbor3D(connolly_points)
        for i, c in enumerate(centers):
            if len(nn.get_in_ball(c, _CONNOLLY_SURFACE_THICKNESS)):
                surface_indices.append(i)
        core_indices = [i for i in range(len(ps)) if i not in surface_indices]

        ps_surface = [ps[i] for i in surface_indices]
        ps_core = [ps[i] for i in core_indices]

        # 6. write BILD script for ChimeraX rendering
        if self.render:
            s = ".color purple\n"
            for p in connolly_points:
                s += ".dot %2.2f %2.2f %2.2f\n" % tuple(p)
            s += "\n"

            # color surface spheres
            s += ".color salmon\n"
            for i in surface_indices:
                center = spheres[i].get_center()
                radius = spheres[i].get_radius() * 1.02
                s += ".sphere %2.2f %2.2f %2.2f %2.2f\n" % (*tuple(center),
                                                            radius)

            # color center indices differently, if they were supplied
            if query_indices:
                s += "\n"
                s += ".color red\n"
                for i in query_indices:
                    center = spheres[i].get_center()
                    radius = spheres[i].get_radius() * 1.04
                    s += ".sphere %2.2f %2.2f %2.2f %2.2f\n" % (*tuple(center),
                                                            radius)
            s += "\n"
            with open("surface.bld", "w") as of:
                of.write(s)

        return ps_surface, ps_core

    
    def get_paratope(self):
        sel = IMP.atom.Selection(
            hierarchy=self.root_hier, 
            resolution=self.resolution,
             molecule=self.ligand_name
        )
        
        if self.paratope_residues:
            sel.set_residue_indexes(self.paratope_residues)
            
        ps = sel.get_selected_particles()
        return ps
    
    
    def get_output(self):
        """
        Overloaded get_output() method of IMP.pmi.restraint.RestraintBase
        that decides what gets output to stat files.
        
        Returns:
        (dict): Dictionary of outputs for stat files.
        """
        
        output = {}
        score = self.evaluate()
        output["TotalScore"] = str(score)
        output["PatchPatchScore_" + self.label] = str(score)
        return output
            
            
                        
class CenterPatchRestraint(RestraintBase):
    def __init__(
        self, root_hier, resolution=1,
        
        receptor_name="receptor",
        ligand_name="ligand",
        
        epitope_center_residue=None,
        paratope_residues=[],
                 
        kappa=1.0, cutoff=8.0,
        
        weight=1.0, label=None
    ):
        
        """
        Distance restraint for the *minimum* distance between an escape mutant 
        residue on the receptor, and a set of residues (usually the CDR3 loop)
        on the ligand.
        
        Args:
        root_hier (IMP.hierarchy): Root hierarchy that provides access to all
        particles in the system.
        
        resolution (float, optional): Coarse grained resolution. Defaults to 1
        residue per bead.
        
        receptor_name (str, optional): Name of receptor molecule in topology.
        Defaults to "receptor".
        
        ligand_name (str, optional): Name of ligand molecule in topology.
        Defaults to "ligand".
        
        epitope_center_residue (int): Target residue on the receptor that
        roughly forms the center of the intended epitope. Defaults to None,
        which triggers a Value Error.
        
        paratope_residues (list, optional): List of residues that form the 
        paratope on the ligand. Defaults to empty list, in which case
        the entire ligand is used as a valid paratope.
        
        kappa (float, optional): Strength of this restraint. Defaults to 1.0.
        
        cutoff (float, optional): Max. allowed value of the closest
        approach distance between receptor and ligand regions. 
        Defaults to 8.0 A.
        
        weight (float, optional): Relative weight of this restraint relative to
        other restraints in the system. Defaults to 1.0.
        
        label (str, optional): Label for this restraint. Defaults to None.
        """
        
        # get epitope center
        if epitope_center_residue is None:
            raise ValueError("Must supply an epitope center residue.")
            
        sel = IMP.atom.Selection(
            root_hier, 
            resolution=resolution, 
            molecule=receptor_name,
            residue_index=epitope_center_residue
        )
        ps_receptor = sel.get_selected_particles()
        assert len(ps_receptor) == 1
        
        # get paratope
        sel = IMP.atom.Selection(
            hierarchy=root_hier,
            resolution=resolution,
            molecule=ligand_name
        )
        
        if paratope_residues:
            sel.set_residue_indexes(paratope_residues)
            
        ps_ligand = sel.get_selected_particles()
        
        # init parent class
        self.model = ps_receptor[0].get_model()
        super().__init__(
            self.model, 
            name=f"CenterCenterScore_{label}", 
            label=label, 
            weight=weight
        )
        
        # initialize a distance pair score with a harmonic upper bound
        ub = IMP.core.HarmonicUpperBound(cutoff, kappa)
        dps = IMP.core.DistancePairScore(ub)
        
        # create a table refiner
        tref = IMP.core.TableRefiner()
        tref.add_particle(ps_receptor[0], ps_receptor)
        tref.add_particle(ps_ligand[0], ps_ligand)
        
        # create closest pair score that wraps the distance pair score
        sf = IMP.core.KClosePairsPairScore(dps, tref, 3)
        
        # create an underlying pair restraint see the implementation in:
        # https://integrativemodeling.org/2.14.0/doc/ref/core_2restrain_minimum_distance_8py-example.html
        pi_pair = (ps_receptor[0], ps_ligand[0])
        restraint = IMP.core.PairRestraint(self.model, sf, pi_pair)
        self.rs.add_restraint(restraint)
        
        self._include_in_rmf = True
    
    
    def get_output(self):
        """
        Overloaded get_output() method of IMP.pmi.restraint.RestraintBase
        that decides what gets output to stat files.
        
        Returns:
        (dict): Dictionary of outputs for stat files.
        """
        
        output = {}
        score = self.evaluate()
        output["_TotalScore"] = str(score)
        output["CenterCenterScore_" + self.label] = str(score)
        return output
    
    