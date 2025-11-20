"""Generated evaluation code for: Late seven-membered ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SevenMemberedRingFormation(BaseScoring):
    """
    Evaluates synthesis routes based on the timing of seven-membered ring formation.
    Rewards late-stage formation of 7-membered rings, which is typically more challenging
    and strategically valuable.
    """
    
    def __init__(self, config: Dict):
        self.ring_size = config.get("ring_size", 7)
        self.timing = config.get("timing", "late")
        self.direction = config.get("direction", "formation")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # Late-stage ring formation is better (closer to 1.0 depth)
            # Convert to 0-10 scale where later formation gets higher score
            return 10 * x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves seven-membered ring formation.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Count 7-membered rings in reactants and products
            reactant_rings = sum(self._count_seven_membered_rings(mol) for mol in reactants)
            product_rings = sum(self._count_seven_membered_rings(mol) for mol in products)
            
            # Check if we're forming a 7-membered ring (more rings in products than reactants)
            return product_rings > reactant_rings
            
        except Exception:
            return False
    
    def _count_seven_membered_rings(self, mol) -> int:
        """
        Count the number of 7-membered rings in a molecule.
        """
        if mol is None:
            return 0
        
        try:
            ring_info = mol.GetRingInfo()
            seven_membered_count = 0
            
            for ring in ring_info.AtomRings():
                if len(ring) == self.ring_size:
                    seven_membered_count += 1
                    
            return seven_membered_count
            
        except Exception:
            return 0
