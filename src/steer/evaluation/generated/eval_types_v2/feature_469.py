"""Generated evaluation code for: Protecting group swap strategy N-benzyl to N-Boc"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupSwap(MultiRxnCondBase):
    """
    Evaluates synthesis routes for protecting group swap strategies.
    Checks if a route contains deprotection of initial protecting group 
    followed by protection with final protecting group on the specified functional group.
    """
    
    def __init__(self, config):
        self.initial_pg = config["initial_pg"].lower()
        self.final_pg = config["final_pg"].lower()
        self.functional_group = config["functional_group"].lower()
        self.swap_present = config["swap_present"]
        
        # Define SMARTS patterns for protecting groups on nitrogen
        self.pg_patterns = {
            "benzyl": "[NH1][CH2]c1ccccc1",  # N-benzyl
            "boc": "[NH1]C(=O)OC(C)(C)C"     # N-Boc
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group operations
        initial_deprotection = False
        final_protection = False
        
        for rxn in reactions:
            if self.detect_deprotection(rxn, self.initial_pg):
                initial_deprotection = True
            if self.detect_protection(rxn, self.final_pg):
                final_protection = True
        
        # Check if swap strategy is present
        swap_detected = initial_deprotection and final_protection
        condition = swap_detected == self.swap_present
        
        return condition, len(reactions)
    
    def detect_deprotection(self, rxn, pg_type):
        """Detect deprotection reaction for specified protecting group"""
        if pg_type not in self.pg_patterns:
            return False
            
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check if protecting group is present in reactants but not in products
        reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
        product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
        
        if not all(reactant_mols) or not all(product_mols):
            return False
            
        pattern = Chem.MolFromSmarts(self.pg_patterns[pg_type])
        if pattern is None:
            return False
            
        # Check if PG present in reactants
        pg_in_reactants = any(mol.HasSubstructMatch(pattern) for mol in reactant_mols if mol)
        
        # Check if PG absent in products (or reduced count)
        pg_in_products = any(mol.HasSubstructMatch(pattern) for mol in product_mols if mol)
        
        return pg_in_reactants and not pg_in_products
    
    def detect_protection(self, rxn, pg_type):
        """Detect protection reaction for specified protecting group"""
        if pg_type not in self.pg_patterns:
            return False
            
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check if protecting group is absent in reactants but present in products
        reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
        product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
        
        if not all(reactant_mols) or not all(product_mols):
            return False
            
        pattern = Chem.MolFromSmarts(self.pg_patterns[pg_type])
        if pattern is None:
            return False
            
        # Check if PG absent in reactants
        pg_in_reactants = any(mol.HasSubstructMatch(pattern) for mol in reactant_mols if mol)
        
        # Check if PG present in products
        pg_in_products = any(mol.HasSubstructMatch(pattern) for mol in product_mols if mol)
        
        return not pg_in_reactants and pg_in_products
