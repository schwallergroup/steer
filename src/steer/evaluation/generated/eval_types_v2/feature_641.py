"""Generated evaluation code for: Protecting group swap strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupSwap(MultiRxnCondBase):
    """
    Evaluates synthesis routes for protecting group swap strategies.
    Detects sequential deprotection of one protecting group followed by
    protection with a different protecting group on the same functional group.
    """
    
    def __init__(self, config):
        self.strategy_type = config.get("strategy_type", "swap")
        self.functional_group = config.get("functional_group", "amine")
        self.from_group = config.get("from_group", "Boc")
        self.to_group = config.get("to_group", "Cbz")
        
        # Define SMARTS patterns for protecting groups
        self.protecting_group_patterns = {
            "Boc": "[NH1]C(=O)OC(C)(C)C",
            "Cbz": "[NH1]C(=O)OCc1ccccc1",
            "Fmoc": "[NH1]C(=O)OCC1c2ccccc2-c2ccccc21",
            "Ac": "[NH1]C(=O)C",
            "Ts": "[NH1]S(=O)(=O)c1ccc(C)cc1",
            "Bn": "[NH2]Cc1ccccc1"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if protecting group swap strategy is present in the route."""
        reactions = self.get_rxns(d)
        
        # Look for sequential deprotection-protection pattern
        swap_found = False
        for i in range(len(reactions) - 1):
            current_rxn = reactions[i]
            next_rxn = reactions[i + 1]
            
            # Check if current reaction removes the from_group
            deprotection = self.detect_deprotection(current_rxn, self.from_group)
            
            # Check if next reaction adds the to_group
            protection = self.detect_protection(next_rxn, self.to_group)
            
            # Verify same functional group is involved
            if deprotection and protection:
                if self.same_functional_group_modified(current_rxn, next_rxn):
                    swap_found = True
                    break
        
        return swap_found, len(reactions)
    
    def detect_deprotection(self, rxn, protecting_group):
        """Detect removal of a specific protecting group."""
        if protecting_group not in self.protecting_group_patterns:
            return False
            
        pattern = Chem.MolFromSmarts(self.protecting_group_patterns[protecting_group])
        if pattern is None:
            return False
        
        # Parse reaction SMILES
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check if protecting group is in reactants but not in products
        pg_in_reactants = False
        pg_in_products = False
        
        for reactant_smiles in reactants:
            mol = Chem.MolFromSmiles(reactant_smiles)
            if mol and mol.HasSubstructMatch(pattern):
                pg_in_reactants = True
                break
        
        for product_smiles in products:
            mol = Chem.MolFromSmiles(product_smiles)
            if mol and mol.HasSubstructMatch(pattern):
                pg_in_products = True
                break
        
        return pg_in_reactants and not pg_in_products
    
    def detect_protection(self, rxn, protecting_group):
        """Detect addition of a specific protecting group."""
        if protecting_group not in self.protecting_group_patterns:
            return False
            
        pattern = Chem.MolFromSmarts(self.protecting_group_patterns[protecting_group])
        if pattern is None:
            return False
        
        # Parse reaction SMILES
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check if protecting group is not in reactants but is in products
        pg_in_reactants = False
        pg_in_products = False
        
        for reactant_smiles in reactants:
            mol = Chem.MolFromSmiles(reactant_smiles)
            if mol and mol.HasSubstructMatch(pattern):
                pg_in_reactants = True
                break
        
        for product_smiles in products:
            mol = Chem.MolFromSmiles(product_smiles)
            if mol and mol.HasSubstructMatch(pattern):
                pg_in_products = True
                break
        
        return not pg_in_reactants and pg_in_products
    
    def same_functional_group_modified(self, rxn1, rxn2):
        """
        Check if the same functional group is modified in both reactions.
        Uses atom mapping to track the same nitrogen atom through both reactions.
        """
        try:
            # For now, simplified check - assumes same molecule core is involved
            # In practice, would use atom mapping to track specific atoms
            rxn1_parts = rxn1.split(">>")
            rxn2_parts = rxn2.split(">>")
            
            # Check if product of first reaction appears as reactant in second
            rxn1_products = set(rxn1_parts[1].split("."))
            rxn2_reactants = set(rxn2_parts[0].split("."))
            
            # Look for overlap indicating sequential reactions on same molecule
            return len(rxn1_products.intersection(rxn2_reactants)) > 0
            
        except:
            return False
