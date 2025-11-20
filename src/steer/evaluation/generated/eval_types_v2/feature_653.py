"""Generated evaluation code for: Late stage quinoline ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageQuinolineFormation(BaseScoring):
    """
    Evaluates routes for late-stage quinoline ring formation.
    Rewards routes where quinoline rings are formed closer to the final product,
    penalizing early formation or absence of quinoline formation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Quinoline formation doesn't happen
        else:
            # Late-stage formation is better (lower depth fraction)
            # Convert to 0-10 scale where late formation gets higher score
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Check if quinoline ring formation occurs in this reaction step.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
                
            # Check if product contains quinoline ring
            product_has_quinoline = product_mol.HasSubstructMatch(self.ring_pattern)
            
            if not product_has_quinoline:
                return False
            
            # Check if any reactant lacks the quinoline ring (indicating formation)
            reactant_smiles_list = reactants_smiles.split(".")
            for reactant_smiles in reactant_smiles_list:
                reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                if reactant_mol and not reactant_mol.HasSubstructMatch(self.ring_pattern):
                    # Found a reactant without quinoline, suggesting ring formation
                    return True
                    
            return False
            
        except Exception:
            return False
