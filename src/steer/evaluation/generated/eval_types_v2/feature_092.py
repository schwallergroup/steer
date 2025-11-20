"""Generated evaluation code for: Late stage cyclopropane formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage ring formation of specified ring size.
    Checks if rings of the target size are formed late in the synthesis route,
    with better scores for later formation.
    """
    
    def __init__(self, config: Dict):
        self.ring_size = config["parameters"]["ring_size"]
        self.timing = config["parameters"]["timing"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # Late-stage formation gets higher score
            # x is depth fraction (0 = early, 1 = late)
            if self.timing == "late":
                return x * 10  # Score 0-10, higher for later formation
            else:
                return (1 - x) * 10  # Score 0-10, higher for earlier formation
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms a ring of the target size."""
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
        if len(rxn) != 2:
            return False
            
        # Parse reactants and product
        reactants_smiles = rxn[0]
        product_smiles = rxn[1]
        
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Count rings of target size in product vs reactants
            product_rings = self._count_rings_of_size(product, self.ring_size)
            reactant_rings = sum(self._count_rings_of_size(r, self.ring_size) for r in reactants)
            
            # Ring formation occurs if product has more rings of target size than reactants
            return product_rings > reactant_rings
            
        except:
            return False
    
    def _count_rings_of_size(self, mol, ring_size: int) -> int:
        """Count the number of rings of specified size in a molecule."""
        if not mol:
            return 0
            
        ring_info = mol.GetRingInfo()
        rings = ring_info.AtomRings()
        
        count = 0
        for ring in rings:
            if len(ring) == ring_size:
                count += 1
                
        return count
