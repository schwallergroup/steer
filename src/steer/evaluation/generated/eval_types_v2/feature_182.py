"""Generated evaluation code for: Early pyrimidine ring formation via condensation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PyrimidineRingFormationTiming(BaseScoring):
    """
    Evaluates the timing of pyrimidine ring formation in synthesis routes.
    Rewards early formation of pyrimidine rings via condensation reactions.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "early"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # For early timing, lower depth fraction is better
            if self.timing == "early":
                return 1 - x  # Early formation gets higher score
            else:
                return x  # Late formation gets higher score
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step forms a pyrimidine ring.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
        
        # Parse reactants and product
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1].split(".")
        
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles if r]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains pyrimidine ring
            product_has_pyrimidine = product.HasSubstructMatch(self.ring_pattern)
            
            # Check if any reactant contains pyrimidine ring
            reactant_has_pyrimidine = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants)
            
            # Ring formation: product has pyrimidine but reactants don't
            if self.direction == "formation":
                return product_has_pyrimidine and not reactant_has_pyrimidine
            # Ring breaking: reactants have pyrimidine but product doesn't
            elif self.direction == "breaking":
                return not product_has_pyrimidine and reactant_has_pyrimidine
            
        except Exception:
            return False
        
        return False
