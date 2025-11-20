"""Generated evaluation code for: Multiple azide to amine conversions"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleAzideReduction(MultiRxnCondBase):
    """
    Evaluates synthesis routes based on multiple azide to amine conversion reactions.
    Checks if the route contains at least the minimum required number of azide reduction reactions.
    """
    
    def __init__(self, config):
        self.minimum_count = config.get("minimum_count", 2)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        azide_reduction_count = sum(1 for r in reactions if self.detect_azide_reduction(r))
        
        condition = azide_reduction_count >= self.minimum_count
        return condition, len(reactions)
    
    def detect_azide_reduction(self, rxn):
        """
        Detects azide to amine reduction reactions by checking for:
        - Azide group ([N-]=[N+]=[N-] or N=[N+]=[N-]) in reactants
        - Primary amine ([NH2]) in products at the same position
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if None in reactants or None in products:
                return False
            
            # Check for azide pattern in reactants
            azide_patterns = [
                Chem.MolFromSmarts("[N-]=[N+]=[N-]"),  # Azide anion
                Chem.MolFromSmarts("N=[N+]=[N-]"),     # Azide neutral form
                Chem.MolFromSmarts("[N-][N+]#N")       # Alternative azide representation
            ]
            
            has_azide_reactant = False
            for reactant in reactants:
                for pattern in azide_patterns:
                    if pattern and reactant.HasSubstructMatch(pattern):
                        has_azide_reactant = True
                        break
                if has_azide_reactant:
                    break
            
            if not has_azide_reactant:
                return False
            
            # Check for primary amine pattern in products
            amine_pattern = Chem.MolFromSmarts("[NH2]")
            has_amine_product = False
            for product in products:
                if amine_pattern and product.HasSubstructMatch(amine_pattern):
                    has_amine_product = True
                    break
            
            return has_azide_reactant and has_amine_product
            
        except Exception:
            return False
