"""Generated evaluation code for: Strategic benzyl protecting group for selectivity"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylProtectingGroupStrategy(BaseScoring):
    """
    Evaluates the strategic use of benzyl protecting groups on nitrogen atoms.
    Checks if benzyl protection is applied to nitrogen before selective reactions,
    particularly useful for preventing competitive N-alkylation during O-alkylation.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not employed
        if self.condition_type == "bool":
            return 1  # Strategy found at any depth
        else:
            # Earlier application of protecting group strategy is better
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d):
        """Check if this reaction involves benzyl protection of nitrogen"""
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Define benzyl group patterns
            benzyl_patterns = [
                "[CH2]c1ccccc1",  # Basic benzyl group
                "Cc1ccccc1",      # Alternative benzyl representation
            ]
            
            # Define N-benzyl patterns (nitrogen bonded to benzyl)
            n_benzyl_patterns = [
                "N[CH2]c1ccccc1",  # Primary or secondary amine with benzyl
                "[NH][CH2]c1ccccc1",  # Secondary amine with benzyl
                "[NH2][CH2]c1ccccc1",  # Primary amine with benzyl (less common representation)
            ]
            
            # Check if benzyl protection is being installed
            # Look for reactions where N-H becomes N-benzyl
            reactant_has_free_n = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts("[NH2]")) or 
                mol.HasSubstructMatch(Chem.MolFromSmarts("[NH1]"))
                for mol in reactant_mols if mol
            )
            
            reactant_has_benzyl_reagent = any(
                any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for pattern in benzyl_patterns)
                for mol in reactant_mols if mol
            )
            
            product_has_n_benzyl = any(
                any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for pattern in n_benzyl_patterns)
                for mol in product_mols if mol
            )
            
            # Check if this is a benzyl protection reaction
            is_benzyl_protection = (
                reactant_has_free_n and 
                reactant_has_benzyl_reagent and 
                product_has_n_benzyl
            )
            
            # Also check for benzyl deprotection (removal strategy)
            reactant_has_n_benzyl = any(
                any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for pattern in n_benzyl_patterns)
                for mol in reactant_mols if mol
            )
            
            product_has_free_n = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts("[NH2]")) or 
                mol.HasSubstructMatch(Chem.MolFromSmarts("[NH1]"))
                for mol in product_mols if mol
            )
            
            is_benzyl_deprotection = reactant_has_n_benzyl and product_has_free_n
            
            return is_benzyl_protection or is_benzyl_deprotection
            
        except Exception:
            return False
