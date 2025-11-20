"""Generated evaluation code for: Early Boc amine protection"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyBocAmineProtection(BaseScoring):
    """
    Evaluates if Boc (tert-butoxycarbonyl) protection of amine groups occurs early in the synthesis.
    Returns higher scores when Boc protection happens at greater depth (earlier in the route).
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No Boc protection found
        # Higher depth (earlier protection) gives better score
        return x * 10
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves Boc protection of an amine."""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if any(mol is None for mol in reactants + products):
                return False
            
            # Check for Boc reagent in reactants (Boc2O or Boc-Cl patterns)
            boc_reagent_patterns = [
                "[CH3][C]([CH3])([CH3])[O][C](=[O])[O][C](=[O])[O][C]([CH3])([CH3])[CH3]",  # Boc2O
                "[CH3][C]([CH3])([CH3])[O][C](=[O])[Cl]"  # Boc-Cl
            ]
            
            has_boc_reagent = False
            for pattern in boc_reagent_patterns:
                pattern_mol = Chem.MolFromSmarts(pattern)
                if pattern_mol and any(reactant.HasSubstructMatch(pattern_mol) for reactant in reactants):
                    has_boc_reagent = True
                    break
            
            if not has_boc_reagent:
                return False
            
            # Check for free amine in reactants and Boc-protected amine in products
            free_amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")  # Primary or secondary amine
            boc_protected_pattern = Chem.MolFromSmarts("[NH1,NH0][C](=[O])[O][C]([CH3])([CH3])[CH3]")  # Boc-protected amine
            
            has_free_amine_reactant = any(reactant.HasSubstructMatch(free_amine_pattern) for reactant in reactants)
            has_boc_protected_product = any(product.HasSubstructMatch(boc_protected_pattern) for product in products)
            
            return has_boc_reagent and has_free_amine_reactant and has_boc_protected_product
            
        except Exception:
            return False
