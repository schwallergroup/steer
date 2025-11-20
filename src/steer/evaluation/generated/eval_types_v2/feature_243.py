"""Generated evaluation code for: Protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates protecting group cycling strategies in synthesis routes.
    
    Checks for specific protecting group transformations like methyl ether 
    to benzyl ether exchange through demethylation followed by benzylation.
    """
    
    def __init__(self, config):
        self.sequence_type = config.get("sequence_type", "exchange")
        self.from_group = config.get("from_group", "methyl_ether")
        self.to_group = config.get("to_group", "benzyl_ether")
        self.steps_count = config.get("steps_count", 2)
        
        # Define SMARTS patterns for protecting groups
        self.protecting_group_patterns = {
            "methyl_ether": "[OH1]-[CH3]",
            "benzyl_ether": "[OH1]-[CH2]-c1ccccc1",
            "tert_butyl_ether": "[OH1]-C(C)(C)C",
            "silyl_ether": "[OH1]-[Si]",
            "acetate": "[OH1]-C(=O)-[CH3]",
            "boc": "[NH1]-C(=O)-O-C(C)(C)C",
            "cbz": "[NH1]-C(=O)-O-[CH2]-c1ccccc1"
        }
    
    def condition_depth(self, d) -> tuple:
        """Check if the protecting group strategy is present in the route."""
        reactions = self.get_rxns(d)
        
        if self.sequence_type == "exchange":
            condition = self._detect_protecting_group_exchange(reactions)
        elif self.sequence_type == "install_remove":
            condition = self._detect_install_remove_cycle(reactions)
        else:
            condition = False
            
        return condition, len(reactions)
    
    def _detect_protecting_group_exchange(self, reactions) -> bool:
        """Detect exchange of one protecting group for another."""
        deprotection_found = False
        protection_found = False
        
        for rxn in reactions:
            # Check for removal of from_group
            if self._detect_deprotection(rxn, self.from_group):
                deprotection_found = True
            
            # Check for installation of to_group  
            if self._detect_protection(rxn, self.to_group):
                protection_found = True
        
        return deprotection_found and protection_found
    
    def _detect_install_remove_cycle(self, reactions) -> bool:
        """Detect installation followed by removal of same protecting group."""
        install_count = 0
        remove_count = 0
        
        for rxn in reactions:
            if self._detect_protection(rxn, self.from_group):
                install_count += 1
            if self._detect_deprotection(rxn, self.from_group):
                remove_count += 1
                
        return install_count >= 1 and remove_count >= 1
    
    def _detect_protection(self, rxn, group_type) -> bool:
        """Check if reaction installs a protecting group."""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(s) for s in rxn_parts[0].split(".") if s]
        products = [Chem.MolFromSmiles(s) for s in rxn_parts[1].split(".") if s]
        
        if not all(reactants) or not all(products):
            return False
            
        pattern = self.protecting_group_patterns.get(group_type)
        if not pattern:
            return False
            
        pattern_mol = Chem.MolFromSmarts(pattern)
        if not pattern_mol:
            return False
        
        # Count protecting groups in reactants vs products
        reactant_count = sum(len(mol.GetSubstructMatches(pattern_mol)) 
                           for mol in reactants)
        product_count = sum(len(mol.GetSubstructMatches(pattern_mol)) 
                          for mol in products)
        
        # Protection increases count of protecting groups
        return product_count > reactant_count
    
    def _detect_deprotection(self, rxn, group_type) -> bool:
        """Check if reaction removes a protecting group."""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(s) for s in rxn_parts[0].split(".") if s]
        products = [Chem.MolFromSmiles(s) for s in rxn_parts[1].split(".") if s]
        
        if not all(reactants) or not all(products):
            return False
            
        pattern = self.protecting_group_patterns.get(group_type)
        if not pattern:
            return False
            
        pattern_mol = Chem.MolFromSmarts(pattern)
        if not pattern_mol:
            return False
        
        # Count protecting groups in reactants vs products
        reactant_count = sum(len(mol.GetSubstructMatches(pattern_mol)) 
                           for mol in reactants)
        product_count = sum(len(mol.GetSubstructMatches(pattern_mol)) 
                          for mol in products)
        
        # Deprotection decreases count of protecting groups
        return reactant_count > product_count
