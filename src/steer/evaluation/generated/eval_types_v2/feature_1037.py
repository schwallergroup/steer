"""Generated evaluation code for: Late stage alkene reduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAlkeneReduction(BaseScoring):
    """
    Evaluates whether alkene reduction (tetrahydropyridine to piperidine) occurs late in the synthesis route.
    Returns higher scores when the reduction happens closer to the final product.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Alkene reduction doesn't occur
        else:
            # Higher score for later reduction (closer to 1.0)
            # Scale to 0-10 range with late-stage preference
            return min(10, max(0, x * 12 - 2))
    
    def hit_condition(self, d):
        """Check if this reaction involves alkene reduction from tetrahydropyridine to piperidine"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            # Parse molecules
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".") if r.strip()]
            
            if not prod_mol or not react_mols:
                return False
                
            # Define SMARTS patterns
            tetrahydropyridine_pattern = Chem.MolFromSmarts("[#6]1[#6][#6]=[#7][#6][#6]1")  # 6-membered ring with C=N
            piperidine_pattern = Chem.MolFromSmarts("[#6]1[#6][#6][#7][#6][#6]1")  # Saturated 6-membered ring with N
            
            if not tetrahydropyridine_pattern or not piperidine_pattern:
                return False
                
            # Check if product contains piperidine
            has_piperidine_product = prod_mol.HasSubstructMatch(piperidine_pattern)
            
            # Check if any reactant contains tetrahydropyridine
            has_tetrahydropyridine_reactant = any(
                mol.HasSubstructMatch(tetrahydropyridine_pattern) for mol in react_mols
            )
            
            # Alkene reduction: tetrahydropyridine (reactant) -> piperidine (product)
            return has_tetrahydropyridine_reactant and has_piperidine_product
            
        except Exception:
            return False
