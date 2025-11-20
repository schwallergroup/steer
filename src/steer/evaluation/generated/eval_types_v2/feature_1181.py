"""Generated evaluation code for: Late stage thiol protection with trityl"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageTritylThiolProtection(BaseScoring):
    """
    Evaluates whether trityl protection of a thiol group occurs at a late stage in the synthesis.
    Returns higher scores when the protection reaction happens closer to the end of the route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)  # Default to late stage
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection doesn't occur
        else:
            # Higher score for later stage protection (closer to 1.0)
            if self.condition_type == "bool":
                return 10 if x >= self.target_depth else 0
            else:
                # Continuous scoring - reward late stage protection
                return max(0, 10 * x)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves trityl protection of a thiol group"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants, products = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mol = Chem.MolFromSmiles(products.strip())
            
            if not all(reactant_mols) or not product_mol:
                return False
            
            # Check for trityl reagent in reactants
            trityl_patterns = [
                "[CH]([c]1ccccc1)([c]2ccccc2)[c]3ccccc3",  # Trityl cation
                "Cl[CH]([c]1ccccc1)([c]2ccccc2)[c]3ccccc3",  # Trityl chloride
                "Br[CH]([c]1ccccc1)([c]2ccccc2)[c]3ccccc3"   # Trityl bromide
            ]
            
            has_trityl_reagent = False
            for reactant in reactant_mols:
                for pattern in trityl_patterns:
                    trityl_mol = Chem.MolFromSmarts(pattern)
                    if trityl_mol and reactant.HasSubstructMatch(trityl_mol):
                        has_trityl_reagent = True
                        break
                if has_trityl_reagent:
                    break
            
            if not has_trityl_reagent:
                return False
            
            # Check for thiol to trityl-protected thiol transformation
            # Free thiol pattern
            thiol_pattern = Chem.MolFromSmarts("[SH1]")
            # Trityl-protected thiol pattern
            protected_thiol_pattern = Chem.MolFromSmarts("[S][CH]([c]1ccccc1)([c]2ccccc2)[c]3ccccc3")
            
            # Check if reactants contain free thiol
            has_free_thiol = any(mol.HasSubstructMatch(thiol_pattern) for mol in reactant_mols)
            
            # Check if product contains trityl-protected thiol
            has_protected_thiol = product_mol.HasSubstructMatch(protected_thiol_pattern)
            
            return has_trityl_reagent and has_free_thiol and has_protected_thiol
            
        except Exception:
            return False
