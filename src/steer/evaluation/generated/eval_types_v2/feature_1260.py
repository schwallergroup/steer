"""Generated evaluation code for: Late stage cyclopropane formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates late-stage cyclopropane formation in synthesis routes.
    Rewards cyclopropane formation reactions that occur at shallow depths (late in synthesis).
    """
    
    def __init__(self, config: Dict):
        self.ring_size = config.get("ring_size", 3)
        self.target_depth = config.get("depth", 1)
        self.reaction_type = config.get("reaction_type", "cyclopropanation")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # Late-stage formation is better (lower depth fraction)
            # Scale to 0-10 range, with depth fraction 0 giving score 10
            return 10 * (1 - x)
    
    def hit_condition(self, d):
        """
        Check if this reaction node represents cyclopropane formation.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Count cyclopropane rings in product vs reactants
            product_cyclopropanes = self._count_cyclopropane_rings(product)
            reactant_cyclopropanes = sum(self._count_cyclopropane_rings(r) for r in reactants)
            
            # Check if cyclopropane rings were formed (net increase)
            return product_cyclopropanes > reactant_cyclopropanes
            
        except Exception:
            return False
    
    def _count_cyclopropane_rings(self, mol):
        """
        Count the number of cyclopropane rings in a molecule.
        """
        if not mol:
            return 0
            
        # SMARTS pattern for cyclopropane ring
        cyclopropane_pattern = Chem.MolFromSmarts("[C;R1]1[C;R1][C;R1]1")
        if not cyclopropane_pattern:
            return 0
            
        matches = mol.GetSubstructMatches(cyclopropane_pattern)
        return len(matches)
