"""Generated evaluation code for: Late pyrimidine ring formation via convergent coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateConvergentPyrimidineFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage pyrimidine ring formation via convergent coupling.
    Checks if a pyrimidine ring is formed through convergent fragment coupling at a late stage
    in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.formation_method = config["parameters"]["formation_method"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen via convergent coupling
        else:
            # For late-stage formation, higher depth fraction is better
            # Score increases as depth approaches 1 (later in synthesis)
            return 10 * x
    
    def hit_condition(self, d):
        """
        Check if this reaction forms a pyrimidine ring via convergent coupling.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains pyrimidine ring
            if not product.HasSubstructMatch(self.ring_pattern):
                return False
            
            # Check if pyrimidine ring is formed (not present in any reactant)
            pyrimidine_in_reactants = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants)
            if pyrimidine_in_reactants:
                return False
            
            # Check for convergent coupling (at least 2 substantial reactants)
            substantial_reactants = [r for r in reactants if r.GetNumHeavyAtoms() >= 3]
            if len(substantial_reactants) < 2:
                return False
            
            # Verify that fragments from different reactants contribute to ring formation
            return self._verify_convergent_ring_formation(reactants, product)
            
        except Exception:
            return False
    
    def _verify_convergent_ring_formation(self, reactants, product):
        """
        Verify that the pyrimidine ring is formed by joining atoms from different reactants.
        Uses atom mapping to trace which reactant atoms contribute to the ring.
        """
        try:
            # Get pyrimidine ring atoms in product
            ring_match = product.GetSubstructMatch(self.ring_pattern)
            if not ring_match:
                return False
            
            # Get atom map numbers for ring atoms
            ring_atom_maps = []
            for atom_idx in ring_match:
                atom = product.GetAtomWithIdx(atom_idx)
                if atom.GetAtomMapNum() > 0:
                    ring_atom_maps.append(atom.GetAtomMapNum())
            
            if len(ring_atom_maps) < 3:  # Need sufficient mapped atoms
                return False
            
            # Check which reactants contain these mapped atoms
            reactant_contributions = set()
            for i, reactant in enumerate(reactants):
                reactant_maps = {atom.GetAtomMapNum() for atom in reactant.GetAtoms() 
                               if atom.GetAtomMapNum() > 0}
                if any(map_num in reactant_maps for map_num in ring_atom_maps):
                    reactant_contributions.add(i)
            
            # True convergent coupling requires contributions from multiple reactants
            return len(reactant_contributions) >= 2
            
        except Exception:
            return False
