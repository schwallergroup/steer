"""Generated evaluation code for: Early stage Grignard reagent preparation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyGrignardPreparation(BaseScoring):
    """
    Evaluates whether Grignard reagent preparation occurs in early stages of synthesis.
    Checks for formation of C-Mg bonds at shallow depths in the synthesis tree.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 7)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Grignard formation doesn't occur
        else:
            # Early stage is better - score decreases with depth
            normalized_depth = min(x, 1.0)  # Cap at 1.0
            return max(0, 1.0 - normalized_depth)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves Grignard reagent formation"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Check for Grignard formation: Mg present in products but not reactants
            # or C-Mg bond formation
            has_mg_reactant = any(self._contains_magnesium(mol) for mol in reactants)
            has_mg_product = any(self._contains_magnesium(mol) for mol in products)
            
            # Grignard formation: Mg appears in products (reagent formation)
            # or organometallic C-Mg bond is formed
            if not has_mg_reactant and has_mg_product:
                return True
            
            # Check for specific Grignard reagent patterns in products
            grignard_pattern = Chem.MolFromSmarts("[C]-[Mg]")
            if grignard_pattern:
                for product in products:
                    if product.HasSubstructMatch(grignard_pattern):
                        # Verify this C-Mg bond wasn't present in reactants
                        reactant_has_grignard = any(
                            reactant.HasSubstructMatch(grignard_pattern) 
                            for reactant in reactants
                        )
                        if not reactant_has_grignard:
                            return True
            
            return False
            
        except Exception:
            return False
    
    def _contains_magnesium(self, mol) -> bool:
        """Check if molecule contains magnesium atom"""
        if mol is None:
            return False
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "Mg":
                return True
        return False
