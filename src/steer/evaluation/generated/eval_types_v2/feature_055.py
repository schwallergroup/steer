"""Generated evaluation code for: Late stage cyclic acetal formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAcetalFormation(BaseScoring):
    """
    Evaluates whether cyclic acetal formation (specifically dioxolane ring formation) 
    occurs in the late stages of synthesis via detection of [O]C[O] substructure formation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config.get("ring_smarts", "[O]C[O]")
        self.timing = config.get("timing", "late")
        self.direction = config.get("direction", "formation")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        elif self.timing == "late":
            return 1 - x  # Late-stage formation is rewarded (closer to 1.0)
        else:
            return x  # Early-stage formation is rewarded
            
    def hit_condition(self, d) -> bool:
        """
        Check if cyclic acetal formation occurs in this reaction step.
        Looks for formation of dioxolane ring pattern [O]C[O].
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if None in reactants or None in products:
                return False
                
            # Create substructure pattern for cyclic acetal
            acetal_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if acetal_pattern is None:
                return False
            
            # Check for ring formation: pattern absent in reactants but present in products
            reactants_have_pattern = any(mol.HasSubstructMatch(acetal_pattern) for mol in reactants if mol is not None)
            products_have_pattern = any(mol.HasSubstructMatch(acetal_pattern) for mol in products if mol is not None)
            
            if self.direction == "formation":
                # Ring formation: not in reactants but in products
                return not reactants_have_pattern and products_have_pattern
            elif self.direction == "breaking":
                # Ring breaking: in reactants but not in products  
                return reactants_have_pattern and not products_have_pattern
            else:
                # Just check presence in products
                return products_have_pattern
                
        except Exception:
            return False
