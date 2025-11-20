"""Generated evaluation code for: Protecting group swap Boc to SEM"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocToSEMSwap(BaseScoring):
    """
    Checks for protecting group swap from Boc to SEM on nitrogen heteroatoms.
    Detects sequential Boc deprotection followed by SEM protection.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "numeric")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not found
        if self.condition_type == "bool":
            return 1  # Strategy found
        else:
            # Earlier strategy is better (lower depth)
            return max(0, 1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves Boc to SEM protecting group swap.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants_smiles, products_smiles = mapped_rxn.split(">>")
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
                
            # Check for Boc deprotection pattern
            boc_deprotection = self._check_boc_deprotection(reactants, products)
            
            # Check for SEM protection pattern  
            sem_protection = self._check_sem_protection(reactants, products)
            
            # Return True if either pattern is found (strategy can span multiple steps)
            return boc_deprotection or sem_protection
            
        except Exception:
            return False
    
    def _check_boc_deprotection(self, reactants, products):
        """Check if Boc group is being removed from nitrogen."""
        # Boc group pattern: tert-butoxycarbonyl
        boc_pattern = Chem.MolFromSmarts("[NH1,NH0]-C(=O)-O-C(C)(C)C")
        free_nh_pattern = Chem.MolFromSmarts("[NH2,NH1]")
        
        if not boc_pattern:
            return False
            
        # Check if reactant has Boc-protected nitrogen
        has_boc_reactant = any(mol.HasSubstructMatch(boc_pattern) for mol in reactants if mol)
        
        # Check if product has free nitrogen (or different protection)
        has_free_n_product = any(mol.HasSubstructMatch(free_nh_pattern) for mol in products if mol)
        
        return has_boc_reactant and has_free_n_product
    
    def _check_sem_protection(self, reactants, products):
        """Check if SEM group is being added to nitrogen."""
        # SEM group pattern: 2-(trimethylsilyl)ethoxymethyl
        sem_pattern = Chem.MolFromSmarts("[NH0,NH1]-C-O-C-C-[Si](C)(C)C")
        free_nh_pattern = Chem.MolFromSmarts("[NH2,NH1]")
        
        if not sem_pattern:
            return False
            
        # Check if reactant has free nitrogen
        has_free_n_reactant = any(mol.HasSubstructMatch(free_nh_pattern) for mol in reactants if mol)
        
        # Check if product has SEM-protected nitrogen
        has_sem_product = any(mol.HasSubstructMatch(sem_pattern) for mol in products if mol)
        
        return has_free_n_reactant and has_sem_product
