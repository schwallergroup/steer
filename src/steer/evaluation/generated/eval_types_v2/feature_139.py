"""Generated evaluation code for: Boc protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates whether a Boc protecting group cycling strategy is used in the synthesis route.
    Checks for both installation and removal of Boc groups, representing temporary protection.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "Boc")
        self.install_and_remove = config.get("install_and_remove", True)
        self.temporary = config.get("temporary", True)
        
        # Boc group SMARTS pattern
        self.boc_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)N")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if Boc protection cycling occurs in the synthesis route."""
        reactions = self.get_rxns(d)
        
        boc_installation = False
        boc_removal = False
        
        for rxn in reactions:
            if self.detect_boc_installation(rxn):
                boc_installation = True
            if self.detect_boc_removal(rxn):
                boc_removal = True
        
        # Check if both installation and removal occur (cycling strategy)
        if self.install_and_remove and self.temporary:
            condition = boc_installation and boc_removal
        elif self.install_and_remove:
            condition = boc_installation and boc_removal
        else:
            condition = boc_installation or boc_removal
            
        return condition, len(reactions)
    
    def detect_boc_installation(self, rxn):
        """Detect if Boc group is installed in this reaction."""
        try:
            reactants_smiles, products_smiles = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Count Boc groups in reactants and products
            reactant_boc_count = sum(len(mol.GetSubstructMatches(self.boc_pattern)) 
                                   for mol in reactants if mol is not None)
            product_boc_count = sum(len(mol.GetSubstructMatches(self.boc_pattern)) 
                                  for mol in products if mol is not None)
            
            # Boc installation: more Boc groups in products than reactants
            return product_boc_count > reactant_boc_count
            
        except Exception:
            return False
    
    def detect_boc_removal(self, rxn):
        """Detect if Boc group is removed in this reaction."""
        try:
            reactants_smiles, products_smiles = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Count Boc groups in reactants and products
            reactant_boc_count = sum(len(mol.GetSubstructMatches(self.boc_pattern)) 
                                   for mol in reactants if mol is not None)
            product_boc_count = sum(len(mol.GetSubstructMatches(self.boc_pattern)) 
                                  for mol in products if mol is not None)
            
            # Boc removal: fewer Boc groups in products than reactants
            return reactant_boc_count > product_boc_count
            
        except Exception:
            return False
