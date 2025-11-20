"""Generated evaluation code for: Late thiazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateThiazoleRingFormation(BaseScoring):
    """
    Evaluates whether thiazole ring formation occurs late in the synthesis route.
    
    Checks for the formation of thiazole rings (c1scnc1) and penalizes if it occurs
    too early in the synthesis. Late-stage ring formation is preferred.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1scnc1"
        self.timing = config["parameters"]["timing"]  # "late"
        self.stage_threshold = config["parameters"]["stage_threshold"]  # 0.8
        self.thiazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            # Penalize early ring formation, reward late formation
            if x >= self.stage_threshold:
                return 10  # Perfect score for very late formation
            else:
                # Linear penalty for early formation
                return x * 10 / self.stage_threshold
        else:
            # For other timing preferences, use inverse scoring
            return (1 - x) * 10
    
    def hit_condition(self, d):
        """
        Check if this reaction step forms a thiazole ring.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            if None in reactants or None in products:
                return False
            
            # Count thiazole rings in reactants and products
            reactant_thiazole_count = sum(
                len(mol.GetSubstructMatches(self.thiazole_pattern)) 
                for mol in reactants if mol is not None
            )
            
            product_thiazole_count = sum(
                len(mol.GetSubstructMatches(self.thiazole_pattern)) 
                for mol in products if mol is not None
            )
            
            # Ring formation occurred if product has more thiazole rings than reactants
            return product_thiazole_count > reactant_thiazole_count
            
        except Exception:
            return False
