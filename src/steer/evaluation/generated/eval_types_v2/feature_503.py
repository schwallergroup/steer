"""Generated evaluation code for: Silicon masking group strategy for alcohol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SiliconMaskingStrategy(BaseScoring):
    """
    Evaluates silicon masking group strategy for alcohols.
    Checks if silylmethyl groups are used as masked hydroxyl groups that can be 
    converted to alcohols via Fleming-Tamao oxidation.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        
        # SMARTS patterns for silicon-containing protecting groups
        self.silyl_patterns = [
            "[Si]C[OH]",  # Direct silylmethyl alcohol
            "[Si]C[CH2][OH]",  # Extended silylmethyl alcohol
            "[Si][CH2]",  # General silylmethyl pattern
            "[Si]([CH3])([CH3])[CH2]",  # TMS-methyl pattern
            "[Si]([CH3])([CH3])([CH3])",  # TMS pattern near carbon chain
        ]
        
        # Pattern for Fleming-Tamao oxidation products (alcohols)
        self.alcohol_pattern = "[CH2][OH]"
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10)"""
        if x < 0:
            return 0  # Strategy not found
        
        if self.condition_type == "bool":
            return 10 if x >= 0 else 0
        else:
            # Earlier use of masking strategy is generally better
            return max(0, 10 * (1 - abs(x - self.target_depth)))
    
    def hit_condition(self, d) -> bool:
        """Check if a reaction involves silicon masking group chemistry"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            products = [mol for mol in products if mol is not None]
            reactants = [mol for mol in reactants if mol is not None]
            
            if not products or not reactants:
                return False
            
            # Check for silyl group introduction (protection)
            silyl_protection = self._check_silyl_protection(reactants, products)
            
            # Check for Fleming-Tamao oxidation (deprotection)
            fleming_tamao = self._check_fleming_tamao_oxidation(reactants, products)
            
            return silyl_protection or fleming_tamao
            
        except Exception:
            return False
    
    def _check_silyl_protection(self, reactants, products) -> bool:
        """Check if reaction introduces silyl protecting groups"""
        # Look for alcohol in reactants and silyl group in products
        has_reactant_alcohol = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts("[OH]")) 
            for mol in reactants
        )
        
        has_product_silyl = any(
            any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                for pattern in self.silyl_patterns)
            for mol in products
        )
        
        return has_reactant_alcohol and has_product_silyl
    
    def _check_fleming_tamao_oxidation(self, reactants, products) -> bool:
        """Check if reaction converts silylmethyl to alcohol (Fleming-Tamao)"""
        # Look for silyl groups in reactants and alcohols in products
        has_reactant_silyl = any(
            any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                for pattern in self.silyl_patterns)
            for mol in reactants
        )
        
        has_product_alcohol = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.alcohol_pattern))
            for mol in products
        )
        
        # Additional check for oxidation conditions (common Fleming-Tamao reagents)
        oxidation_reagents = ["[H][O][O][H]", "[O]", "O=O"]  # H2O2, oxone, O2
        has_oxidant = any(
            any(mol.HasSubstructMatch(Chem.MolFromSmarts(ox_pattern))
                for ox_pattern in oxidation_reagents)
            for mol in reactants
        )
        
        return has_reactant_silyl and has_product_alcohol and has_oxidant
