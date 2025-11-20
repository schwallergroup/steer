"""Generated evaluation code for: Late stage pteridine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PteridineRingFormation(BaseScoring):
    """
    Evaluates the timing of pteridine ring formation in synthesis routes.
    Rewards late-stage formation of the pteridine core structure.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Later formation gets higher score
            else:
                return x  # Earlier formation gets higher score
    
    def hit_condition(self, d):
        """
        Check if pteridine ring is formed in this reaction step.
        Ring formation occurs when product has the ring but reactants don't.
        """
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        product = Chem.MolFromSmiles(rxn_parts[0])
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".")]
        
        # Remove None molecules (parsing failures)
        reactants = [r for r in reactants if r is not None]
        
        if product is None or not reactants:
            return False
        
        # Check if product contains pteridine ring
        product_has_ring = product.HasSubstructMatch(self.ring_pattern)
        
        if not product_has_ring:
            return False
        
        # Check if any reactant already contains the complete pteridine ring
        reactants_have_ring = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants)
        
        # Ring formation occurs when product has ring but no reactant has complete ring
        if self.direction == "formation":
            return product_has_ring and not reactants_have_ring
        else:  # ring breaking
            return not product_has_ring and reactants_have_ring
