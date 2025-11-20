"""Generated evaluation code for: Protecting group cycling on same alcohol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes for protecting group cycling on the same alcohol position.
    Detects when an alcohol is deprotected and then immediately reprotected 
    with a different protecting group at the same position.
    """
    
    def __init__(self, config):
        self.functional_group = config.get("functional_group", "alcohol")
        self.strategy = config.get("strategy", "cycling")
        self.groups = config.get("groups", ["acetate", "TMS"])
        self.same_position = config.get("same_position", True)
        
        # Define protecting group patterns
        self.protecting_patterns = {
            "acetate": "[OH1:1]>>O([C:1])C(=O)C",  # Acetate protection
            "TMS": "[OH1:1]>>O([C:1])[Si](C)(C)C"  # TMS protection
        }
        
        # Define deprotection patterns (reverse of protection)
        self.deprotecting_patterns = {
            "acetate": "O([C:1])C(=O)C>>[OH1:1]",  # Acetate deprotection
            "TMS": "O([C:1])[Si](C)(C)C>>[OH1:1]"  # TMS deprotection
        }

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        cycling_detected = self.detect_protecting_group_cycling(reactions)
        return cycling_detected, len(reactions)

    def detect_protecting_group_cycling(self, reactions) -> bool:
        """
        Detect if there's protecting group cycling on the same alcohol position.
        """
        for i in range(len(reactions) - 1):
            current_rxn = reactions[i]
            next_rxn = reactions[i + 1]
            
            # Check if current reaction is deprotection and next is protection
            if self.is_deprotection(current_rxn) and self.is_protection(next_rxn):
                if self.same_position:
                    # Check if it's the same alcohol position being modified
                    if self.same_alcohol_position(current_rxn, next_rxn):
                        # Check if different protecting groups are used
                        deprotected_group = self.get_protecting_group_type(current_rxn, is_deprotection=True)
                        protected_group = self.get_protecting_group_type(next_rxn, is_deprotection=False)
                        
                        if (deprotected_group in self.groups and 
                            protected_group in self.groups and 
                            deprotected_group != protected_group):
                            return True
        return False

    def is_protection(self, rxn) -> bool:
        """Check if reaction involves protection of alcohol."""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = Chem.MolFromSmiles(rxn_parts[0])
        products = Chem.MolFromSmiles(rxn_parts[1])
        
        if reactants is None or products is None:
            return False
        
        # Check for alcohol in reactants and protected alcohol in products
        alcohol_pattern = Chem.MolFromSmarts("[OH1]")
        has_free_alcohol_reactant = reactants.HasSubstructMatch(alcohol_pattern)
        
        # Check for any protecting group patterns in products
        for group in self.groups:
            if group in self.protecting_patterns:
                # Create pattern for protected alcohol
                if group == "acetate":
                    protected_pattern = Chem.MolFromSmarts("OC(=O)C")
                elif group == "TMS":
                    protected_pattern = Chem.MolFromSmarts("O[Si](C)(C)C")
                else:
                    continue
                    
                if products.HasSubstructMatch(protected_pattern):
                    return has_free_alcohol_reactant
        
        return False

    def is_deprotection(self, rxn) -> bool:
        """Check if reaction involves deprotection of alcohol."""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = Chem.MolFromSmiles(rxn_parts[0])
        products = Chem.MolFromSmiles(rxn_parts[1])
        
        if reactants is None or products is None:
            return False
        
        # Check for protected alcohol in reactants and free alcohol in products
        alcohol_pattern = Chem.MolFromSmarts("[OH1]")
        has_free_alcohol_product = products.HasSubstructMatch(alcohol_pattern)
        
        # Check for any protecting group patterns in reactants
        for group in self.groups:
            if group == "acetate":
                protected_pattern = Chem.MolFromSmarts("OC(=O)C")
            elif group == "TMS":
                protected_pattern = Chem.MolFromSmarts("O[Si](C)(C)C")
            else:
                continue
                
            if reactants.HasSubstructMatch(protected_pattern):
                return has_free_alcohol_product
        
        return False

    def get_protecting_group_type(self, rxn, is_deprotection=False) -> str:
        """Identify the type of protecting group involved in the reaction."""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return ""
        
        mol_to_check = Chem.MolFromSmiles(rxn_parts[0] if is_deprotection else rxn_parts[1])
        if mol_to_check is None:
            return ""
        
        for group in self.groups:
            if group == "acetate":
                pattern = Chem.MolFromSmarts("OC(=O)C")
            elif group == "TMS":
                pattern = Chem.MolFromSmarts("O[Si](C)(C)C")
            else:
                continue
                
            if mol_to_check.HasSubstructMatch(pattern):
                return group
        
        return ""

    def same_alcohol_position(self, deprotection_rxn, protection_rxn) -> bool:
        """
        Check if the same alcohol position is involved in consecutive reactions.
        Uses atom mapping to track the same carbon atom.
        """
        try:
            # Parse deprotection reaction
            deprotect_parts = deprotection_rxn.split(">>")
            deprotect_reactant = Chem.MolFromSmiles(deprotect_parts[0])
            deprotect_product = Chem.MolFromSmiles(deprotect_parts[1])
            
            # Parse protection reaction  
            protect_parts = protection_rxn.split(">>")
            protect_reactant = Chem.MolFromSmiles(protect_parts[0])
            protect_product = Chem.MolFromSmiles(protect_parts[1])
            
            if None in [deprotect_reactant, deprotect_product, protect_reactant, protect_product]:
                return False
            
            # Find alcohol carbons with atom mapping
            deprotect_alcohol_maps = set()
            protect_alcohol_maps = set()
            
            # Get atom map numbers for alcohols in deprotection product
            for atom in deprotect_product.GetAtoms():
                if atom.GetAtomMapNum() > 0:
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'O' and neighbor.GetTotalDegree() == 1:
                            deprotect_alcohol_maps.add(atom.GetAtomMapNum())
            
            # Get atom map numbers for alcohols in protection reactant
            for atom in protec
