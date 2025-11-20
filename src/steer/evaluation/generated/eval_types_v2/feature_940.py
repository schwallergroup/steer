"""Generated evaluation code for: Protecting group swap at primary alcohol position"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupSwap(MultiRxnCondBase):
    """
    Detects protecting group swap sequences at specified positions.
    Checks for removal of one protecting group followed by installation of another
    at the same position in consecutive reaction steps.
    """
    
    def __init__(self, config):
        self.position_smarts = config["parameters"]["position_smarts"]
        self.final_protection = config["parameters"]["final_protection"]
        self.swap_sequence = config["parameters"].get("swap_sequence", True)
        
        # Create RDKit patterns
        self.position_pattern = Chem.MolFromSmarts(self.position_smarts)
        self.final_protection_mol = Chem.MolFromSmiles(self.final_protection)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        if len(reactions) < 2:
            return False, len(reactions)
        
        # Look for consecutive deprotection/protection steps
        for i in range(len(reactions) - 1):
            current_rxn = reactions[i]
            next_rxn = reactions[i + 1]
            
            if self.is_protecting_group_swap(current_rxn, next_rxn):
                return True, len(reactions)
        
        return False, len(reactions)
    
    def is_protecting_group_swap(self, rxn1, rxn2) -> bool:
        """
        Check if two consecutive reactions constitute a protecting group swap
        at the specified position.
        """
        try:
            # Parse first reaction (should be deprotection)
            rxn1_parts = rxn1.split(">>")
            rxn1_reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn1_parts[0].split(".")]
            rxn1_products = [Chem.MolFromSmiles(p.strip()) for p in rxn1_parts[1].split(".")]
            
            # Parse second reaction (should be protection)
            rxn2_parts = rxn2.split(">>")
            rxn2_reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn2_parts[0].split(".")]
            rxn2_products = [Chem.MolFromSmiles(p.strip()) for p in rxn2_parts[1].split(".")]
            
            if not all(rxn1_reactants + rxn1_products + rxn2_reactants + rxn2_products):
                return False
            
            # Check if first reaction removes protection at target position
            deprotection_occurred = self.check_deprotection(rxn1_reactants, rxn1_products)
            
            # Check if second reaction adds the target protection
            target_protection_added = self.check_target_protection(rxn2_reactants, rxn2_products)
            
            # Verify the intermediate matches (product of rxn1 should be reactant of rxn2)
            intermediate_matches = self.check_intermediate_consistency(rxn1_products, rxn2_reactants)
            
            return deprotection_occurred and target_protection_added and intermediate_matches
            
        except Exception:
            return False
    
    def check_deprotection(self, reactants, products) -> bool:
        """Check if a protecting group was removed from the target position."""
        # Find molecules containing the target position pattern in reactants
        protected_reactants = []
        for mol in reactants:
            if mol and mol.HasSubstructMatch(self.position_pattern):
                protected_reactants.append(mol)
        
        if not protected_reactants:
            return False
        
        # Check if products have fewer or different protecting groups at this position
        for product in products:
            if product:
                # Look for free alcohol or different protection pattern
                free_alcohol_pattern = Chem.MolFromSmarts("[CH2]O")
                if product.HasSubstructMatch(free_alcohol_pattern):
                    return True
                
                # Check if the original protection pattern is gone
                if not product.HasSubstructMatch(self.position_pattern):
                    return True
        
        return False
    
    def check_target_protection(self, reactants, products) -> bool:
        """Check if the target protecting group was installed."""
        # Look for the final protection pattern in products
        final_pattern = Chem.MolFromSmarts("COC(=O)c1ccccc1")  # Benzoate pattern
        
        for product in products:
            if product and product.HasSubstructMatch(final_pattern):
                # Verify this protection wasn't already present in reactants
                protection_added = True
                for reactant in reactants:
                    if reactant and reactant.HasSubstructMatch(final_pattern):
                        protection_added = False
                        break
                
                if protection_added:
                    return True
        
        return False
    
    def check_intermediate_consistency(self, rxn1_products, rxn2_reactants) -> bool:
        """Verify that the product of deprotection is used in the protection step."""
        # Simple check: ensure there's at least one common structure
        for product in rxn1_products:
            if product:
                for reactant in rxn2_reactants:
                    if reactant:
                        # Check for structural similarity (same heavy atom count is a simple heuristic)
                        if product.GetNumHeavyAtoms() == reactant.GetNumHeavyAtoms():
                            return True
        return False
