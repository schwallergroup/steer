"""Generated evaluation code for: Boc protecting group strategy for lactam nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocLactamProtection(BaseScoring):
    """
    Evaluates Boc protecting group strategy for lactam nitrogen.
    Checks if a Boc group is used to protect lactam nitrogen during synthesis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", 0)
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if x < 0:
                return 0  # Protection strategy not found
            else:
                return 1  # Protection strategy found
        else:
            if x < 0:
                return 0
            return max(0, 1 - abs(x - self.target_depth))
    
    def hit_condition(self, d):
        """
        Check if reaction involves Boc protection of lactam nitrogen
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants_smiles, products_smiles = mapped_rxn.split(">>")
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for Boc protection: lactam N in reactants, Boc-protected lactam N in products
            return self._has_boc_lactam_protection(reactants, products)
            
        except:
            return False
    
    def _has_boc_lactam_protection(self, reactants, products):
        """
        Check if reaction shows Boc protection of lactam nitrogen
        """
        # Lactam patterns (various ring sizes)
        lactam_patterns = [
            "[NH1]C(=O)",  # General lactam pattern
            "[NH1]1CCCC(=O)1",  # 5-membered lactam (pyrrolidinone)
            "[NH1]1CCCCC(=O)1",  # 6-membered lactam (piperidinone)
            "[NH1]1CCCCCC(=O)1",  # 7-membered lactam
        ]
        
        # Boc-protected lactam patterns
        boc_lactam_patterns = [
            "N([CH3]OC(=O)C([CH3])([CH3])[CH3])C(=O)",  # Boc-protected amide
            "N(C(=O)OC(C)(C)C)C(=O)",  # Boc-protected lactam (simplified)
        ]
        
        # Check if reactants contain unprotected lactam
        has_unprotected_lactam = False
        for reactant in reactants:
            for pattern in lactam_patterns:
                lactam_smarts = Chem.MolFromSmarts(pattern)
                if lactam_smarts and reactant.HasSubstructMatch(lactam_smarts):
                    has_unprotected_lactam = True
                    break
            if has_unprotected_lactam:
                break
        
        # Check if products contain Boc-protected lactam
        has_boc_protected_lactam = False
        for product in products:
            for pattern in boc_lactam_patterns:
                boc_smarts = Chem.MolFromSmarts(pattern)
                if boc_smarts and product.HasSubstructMatch(boc_smarts):
                    has_boc_protected_lactam = True
                    break
            if has_boc_protected_lactam:
                break
        
        # Also check for presence of Boc reagent in reactants
        boc_reagent_patterns = [
            "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C",  # Boc2O
            "CC(C)(C)OC(=O)Cl",  # Boc-Cl
        ]
        
        has_boc_reagent = False
        for reactant in reactants:
            for pattern in boc_reagent_patterns:
                boc_reagent_smarts = Chem.MolFromSmarts(pattern)
                if boc_reagent_smarts and reactant.HasSubstructMatch(boc_reagent_smarts):
                    has_boc_reagent = True
                    break
            if has_boc_reagent:
                break
        
        return has_unprotected_lactam and (has_boc_protected_lactam or has_boc_reagent)
