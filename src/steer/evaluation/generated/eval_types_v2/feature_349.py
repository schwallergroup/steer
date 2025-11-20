"""Generated evaluation code for: Late stage heterocyclic core formation via cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageHeterocyclicCyclization(BaseScoring):
    """
    Evaluates if heterocyclic core formation occurs via late-stage intramolecular cyclization.
    Checks for ring-forming reactions that create heterocycles through intramolecular bond formation.
    """
    
    def __init__(self, config: Dict):
        self.ring_count = config["parameters"]["ring_count"]
        self.timing = config["parameters"]["timing"]  # "late", "early", or "any"
        self.formation_type = config["parameters"]["formation_type"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        
        if self.timing == "late":
            # Reward later cyclization (lower depth fraction is better for late timing)
            return (1 - x) * 10
        elif self.timing == "early":
            # Reward earlier cyclization
            return x * 10
        else:  # "any"
            return 10  # Just needs to happen somewhere
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves heterocyclic ring formation via cyclization"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if this is an intramolecular cyclization (single reactant forming ring)
            if self.formation_type == "intramolecular_cyclization" and len(reactants) != 1:
                return False
            
            # Count rings in reactants vs product
            reactant_rings = sum(mol.GetRingInfo().NumRings() for mol in reactants)
            product_rings = product.GetRingInfo().NumRings()
            
            # Check if the expected number of rings were formed
            rings_formed = product_rings - reactant_rings
            if rings_formed != self.ring_count:
                return False
            
            # Check if at least one of the newly formed rings contains heteroatoms
            return self._has_new_heterocycle(reactants[0] if len(reactants) == 1 else None, product)
            
        except Exception:
            return False
    
    def _has_new_heterocycle(self, reactant, product):
        """Check if a new heterocycle was formed"""
        if not reactant:
            # For multiple reactants, just check if product has heterocycles
            return self._contains_heterocycle(product)
        
        # Get ring systems in reactant and product
        reactant_ring_atoms = set()
        for ring in reactant.GetRingInfo().AtomRings():
            reactant_ring_atoms.update(ring)
        
        product_ring_atoms = set()
        product_rings = []
        for ring in product.GetRingInfo().AtomRings():
            product_ring_atoms.update(ring)
            product_rings.append(ring)
        
        # Find newly formed rings (atoms that are in rings in product but not reactant)
        # This is simplified - in practice would need atom mapping for accurate comparison
        
        # Alternative approach: check if product has heterocycles
        # and more rings than reactant
        return (self._contains_heterocycle(product) and 
                product.GetRingInfo().NumRings() > reactant.GetRingInfo().NumRings())
    
    def _contains_heterocycle(self, mol):
        """Check if molecule contains heterocyclic rings"""
        if not mol:
            return False
            
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            # Check if ring contains heteroatoms (N, O, S, etc.)
            for atom_idx in ring:
                atom = mol.GetAtomWithIdx(atom_idx)
                if atom.GetAtomicNum() not in [6, 1]:  # Not C or H
                    return True
        return False
