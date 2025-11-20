"""Generated evaluation code for: Late pyrazine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePyrazineFormation(BaseScoring):
    """
    Evaluates whether pyrazine ring formation occurs late in the synthesis route.
    Checks for the formation of pyrazine rings (c1cnccn1) and rewards later occurrence.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.pyrazine_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # Late-stage ring formation is better (higher depth fraction = better score)
            if self.timing == "late":
                return x * 10  # Convert depth fraction to 0-10 scale, rewarding later formation
            else:
                return (1 - x) * 10  # Early formation would be rewarded
    
    def hit_condition(self, d) -> bool:
        """
        Check if pyrazine ring formation occurs in this reaction step.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            product = Chem.MolFromSmiles(product_smiles)
            if not product:
                return False
                
            # Count pyrazine rings in product
            product_pyrazine_count = len(product.GetSubstructMatches(self.pyrazine_pattern))
            
            # Count pyrazine rings in all reactants
            reactant_pyrazine_count = 0
            for reactant_smiles in reactants_smiles.split("."):
                reactant = Chem.MolFromSmiles(reactant_smiles)
                if reactant:
                    reactant_pyrazine_count += len(reactant.GetSubstructMatches(self.pyrazine_pattern))
            
            # Ring formation occurs if product has more pyrazine rings than reactants
            if self.direction == "formation":
                return product_pyrazine_count > reactant_pyrazine_count
            elif self.direction == "breaking":
                return product_pyrazine_count < reactant_pyrazine_count
                
        except Exception:
            return False
            
        return False
