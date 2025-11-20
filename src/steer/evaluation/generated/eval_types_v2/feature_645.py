"""Generated evaluation code for: Protecting group swap sequence strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupSwap(MultiRxnCondBase):
    """
    Evaluates routes for protecting group swap sequences where one protecting group
    is removed and immediately replaced with another in consecutive reactions.
    """
    
    def __init__(self, config):
        self.sequence_type = config.get("sequence_type", "swap")
        self.groups = config.get("groups", ["Boc", "Cbz"])
        self.consecutive = config.get("consecutive", True)
        
        # Define SMARTS patterns for common protecting groups
        self.protection_patterns = {
            "Boc": "[NX3][CX3](=O)[OX2][CX4]([CH3])([CH3])[CH3]",  # tert-butoxycarbonyl
            "Cbz": "[NX3][CX3](=O)[OX2][CH2][c1ccccc1]",  # benzyloxycarbonyl
            "Fmoc": "[NX3][CX3](=O)[OX2][CH2][CH1]1c2ccccc2c3ccccc13",  # fluorenylmethoxycarbonyl
            "Ts": "[NX3][SX4](=O)(=O)c1ccc([CH3])cc1",  # tosyl
            "Ac": "[NX3][CX3](=O)[CH3]",  # acetyl
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if the route contains the specified protecting group swap sequence."""
        reactions = self.get_rxns(d)
        
        if len(reactions) < 2:
            return False, len(reactions)
        
        swap_found = self._detect_swap_sequence(reactions)
        return swap_found, len(reactions)
    
    def _detect_swap_sequence(self, reactions) -> bool:
        """Detect if the swap sequence occurs in the reactions."""
        if len(self.groups) != 2:
            return False
        
        group1, group2 = self.groups[0], self.groups[1]
        
        for i in range(len(reactions) - 1):
            current_rxn = reactions[i]
            next_rxn = reactions[i + 1]
            
            # Check if current reaction removes group1 and next reaction adds group2
            if (self._is_deprotection(current_rxn, group1) and 
                self._is_protection(next_rxn, group2)):
                
                # If consecutive is required, they must be adjacent reactions
                if self.consecutive:
                    return True
                else:
                    return True
            
            # Also check reverse direction (remove group2, add group1)
            if (self._is_deprotection(current_rxn, group2) and 
                self._is_protection(next_rxn, group1)):
                
                if self.consecutive:
                    return True
                else:
                    return True
        
        return False
    
    def _is_deprotection(self, rxn, protecting_group) -> bool:
        """Check if reaction removes a specific protecting group."""
        if protecting_group not in self.protection_patterns:
            return False
        
        pattern = self.protection_patterns[protecting_group]
        
        try:
            reactants = [Chem.MolFromSmiles(smi) for smi in rxn[0].split('.')]
            products = [Chem.MolFromSmiles(smi) for smi in rxn[1].split('.')]
            
            # Count protecting group occurrences in reactants and products
            reactant_count = sum(self._count_substructures(mol, pattern) for mol in reactants if mol)
            product_count = sum(self._count_substructures(mol, pattern) for mol in products if mol)
            
            # Deprotection means fewer protecting groups in products
            return reactant_count > product_count
            
        except:
            return False
    
    def _is_protection(self, rxn, protecting_group) -> bool:
        """Check if reaction adds a specific protecting group."""
        if protecting_group not in self.protection_patterns:
            return False
        
        pattern = self.protection_patterns[protecting_group]
        
        try:
            reactants = [Chem.MolFromSmiles(smi) for smi in rxn[0].split('.')]
            products = [Chem.MolFromSmiles(smi) for smi in rxn[1].split('.')]
            
            # Count protecting group occurrences in reactants and products
            reactant_count = sum(self._count_substructures(mol, pattern) for mol in reactants if mol)
            product_count = sum(self._count_substructures(mol, pattern) for mol in products if mol)
            
            # Protection means more protecting groups in products
            return product_count > reactant_count
            
        except:
            return False
    
    def _count_substructures(self, mol, pattern_smarts) -> int:
        """Count occurrences of a substructure pattern in a molecule."""
        if mol is None:
            return 0
        
        try:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if pattern is None:
                return 0
            return len(mol.GetSubstructMatches(pattern))
        except:
            return 0
