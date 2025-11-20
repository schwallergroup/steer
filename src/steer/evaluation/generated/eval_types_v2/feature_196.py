"""Generated evaluation code for: Protecting group swap strategy Boc to Cbz"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocToCbzSwap(MultiRxnCondBase):
    """
    Evaluates synthesis routes for Boc to Cbz protecting group swap strategy.
    Checks if the route contains both Boc deprotection and Cbz protection reactions
    in the correct sequence.
    """
    
    def __init__(self, config):
        self.strategy_type = config.get("strategy_type", "swap")
        self.from_group = config.get("from_group", "Boc")
        self.to_group = config.get("to_group", "Cbz")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        boc_deprotection_found = False
        cbz_protection_found = False
        
        # Check reactions in order (later reactions have higher indices)
        for i, rxn in enumerate(reactions):
            if self.detect_boc_deprotection(rxn):
                boc_deprotection_found = True
            elif self.detect_cbz_protection(rxn) and boc_deprotection_found:
                cbz_protection_found = True
                break
        
        # Strategy successful if both steps found in correct order
        condition = boc_deprotection_found and cbz_protection_found
        return condition, len(reactions)
    
    def detect_boc_deprotection(self, rxn):
        """Detect Boc deprotection reaction (Boc group removal from amine)"""
        try:
            reactants_smiles, products_smiles = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Boc group pattern: tert-butyl carbamate
            boc_pattern = Chem.MolFromSmarts("[NH1,NH2]-C(=O)-O-C(C)(C)C")
            free_amine_pattern = Chem.MolFromSmarts("[NH2,NH3+]")
            
            # Check if reactants contain Boc-protected amine
            has_boc_reactant = any(mol and mol.HasSubstructMatch(boc_pattern) for mol in reactants if mol)
            
            # Check if products contain free amine
            has_free_amine_product = any(mol and mol.HasSubstructMatch(free_amine_pattern) for mol in products if mol)
            
            return has_boc_reactant and has_free_amine_product
            
        except Exception:
            return False
    
    def detect_cbz_protection(self, rxn):
        """Detect Cbz protection reaction (benzyl carbamate formation)"""
        try:
            reactants_smiles, products_smiles = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Free amine pattern
            free_amine_pattern = Chem.MolFromSmarts("[NH2,NH3+]")
            
            # Cbz group pattern: benzyl carbamate
            cbz_pattern = Chem.MolFromSmarts("[NH1]-C(=O)-O-Cc1ccccc1")
            
            # Check if reactants contain free amine
            has_free_amine_reactant = any(mol and mol.HasSubstructMatch(free_amine_pattern) for mol in reactants if mol)
            
            # Check if products contain Cbz-protected amine
            has_cbz_product = any(mol and mol.HasSubstructMatch(cbz_pattern) for mol in products if mol)
            
            return has_free_amine_reactant and has_cbz_product
            
        except Exception:
            return False
