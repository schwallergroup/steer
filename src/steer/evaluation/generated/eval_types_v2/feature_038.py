"""Generated evaluation code for: Late stage amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Evaluates whether amide coupling occurs as the final step in the synthesis route.
    Returns higher scores when amide bond formation happens at the very end of the route.
    """
    
    def __init__(self, config: Dict):
        self.position = config.get("position", "final_step")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Amide coupling doesn't happen
        elif x == 1.0:  # Final step (depth fraction = 1.0)
            return 10
        else:
            # Penalize earlier amide coupling - later is better
            return max(0, 10 * x - 5)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves amide bond formation"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Count amide bonds in product and reactants
            amide_pattern = Chem.MolFromSmarts("[C;!$(C=O)][C](=O)[N;!$(N=O)]")
            
            product_amides = len(product.GetSubstructMatches(amide_pattern))
            reactant_amides = sum(len(r.GetSubstructMatches(amide_pattern)) for r in reactants)
            
            # Amide coupling if product has more amide bonds than reactants
            return product_amides > reactant_amides
            
        except Exception:
            return False
