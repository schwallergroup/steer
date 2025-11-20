"""Generated evaluation code for: Sequential protecting group swap strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupSwap(MultiRxnCondBase):
    """
    Evaluates routes that perform sequential protecting group swap strategy.
    Detects deprotection of one group followed by protection with another group
    on the same functional group (e.g., amine).
    """
    
    def __init__(self, config):
        self.protection_groups = config.get("protection_groups", ["Boc", "Moc"])
        self.swap_sequence = config.get("swap_sequence", True)
        self.functional_group = config.get("same_functional_group", "amine")
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "Boc": "[NX3][C](=O)OC(C)(C)C",  # tert-butoxycarbonyl
            "Moc": "[NX3][C](=O)OCC1=CC=CC=C1",  # methoxycarbonyl
            "Fmoc": "[NX3][C](=O)OCC1C2=CC=CC=C2C3=CC=CC=C13",  # fluorenylmethoxycarbonyl
            "Cbz": "[NX3][C](=O)OCC1=CC=CC=C1",  # carbobenzoxy
        }
        
        # Pattern for free amine
        self.free_amine_pattern = "[NH2,NH1]"
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        if len(reactions) < 2:
            return False, len(reactions)
        
        # Look for sequential deprotection-protection pattern
        found_swap = False
        
        for i in range(len(reactions) - 1):
            current_rxn = reactions[i]
            next_rxn = reactions[i + 1]
            
            # Check if current reaction is deprotection of first group
            deprotection = self.detect_deprotection(current_rxn, self.protection_groups[0])
            
            # Check if next reaction is protection with second group
            protection = self.detect_protection(next_rxn, self.protection_groups[1])
            
            if deprotection and protection:
                # Verify it's on the same functional group by checking for free amine intermediate
                if self.has_free_amine_intermediate(current_rxn):
                    found_swap = True
                    break
        
        condition = found_swap == self.swap_sequence
        return condition, len(reactions)
    
    def detect_deprotection(self, rxn, protecting_group):
        """Detect removal of a protecting group"""
        if protecting_group not in self.pg_patterns:
            return False
            
        pattern = self.pg_patterns[protecting_group]
        rxn_parts = rxn.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants = Chem.MolFromSmiles(rxn_parts[0])
        products_smiles = rxn_parts[1].split(".")
        
        if reactants is None:
            return False
            
        # Check if reactant has the protecting group
        pg_mol = Chem.MolFromSmarts(pattern)
        if pg_mol is None or not reactants.HasSubstructMatch(pg_mol):
            return False
        
        # Check if products have free amine
        for prod_smi in products_smiles:
            prod_mol = Chem.MolFromSmiles(prod_smi)
            if prod_mol is not None:
                free_amine_mol = Chem.MolFromSmarts(self.free_amine_pattern)
                if free_amine_mol is not None and prod_mol.HasSubstructMatch(free_amine_mol):
                    return True
        
        return False
    
    def detect_protection(self, rxn, protecting_group):
        """Detect addition of a protecting group"""
        if protecting_group not in self.pg_patterns:
            return False
            
        pattern = self.pg_patterns[protecting_group]
        rxn_parts = rxn.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0].split(".")
        products = Chem.MolFromSmiles(rxn_parts[1])
        
        if products is None:
            return False
            
        # Check if product has the protecting group
        pg_mol = Chem.MolFromSmarts(pattern)
        if pg_mol is None or not products.HasSubstructMatch(pg_mol):
            return False
        
        # Check if one of the reactants has free amine
        for react_smi in reactants_smiles:
            react_mol = Chem.MolFromSmiles(react_smi)
            if react_mol is not None:
                free_amine_mol = Chem.MolFromSmarts(self.free_amine_pattern)
                if free_amine_mol is not None and react_mol.HasSubstructMatch(free_amine_mol):
                    return True
        
        return False
    
    def has_free_amine_intermediate(self, rxn):
        """Check if the deprotection reaction produces a free amine"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        products_smiles = rxn_parts[1].split(".")
        
        for prod_smi in products_smiles:
            prod_mol = Chem.MolFromSmiles(prod_smi)
            if prod_mol is not None:
                free_amine_mol = Chem.MolFromSmarts(self.free_amine_pattern)
                if free_amine_mol is not None and prod_mol.HasSubstructMatch(free_amine_mol):
                    return True
        
        return False
