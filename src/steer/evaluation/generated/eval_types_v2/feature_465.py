"""Generated evaluation code for: Boc protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes based on Boc protecting group cycling strategy.
    Checks if the route involves deprotecting and re-protecting the same amine 
    with Boc groups in a cyclic manner.
    """
    
    def __init__(self, config):
        self.deprotection_reprotection = config.get("deprotection_reprotection", True)
        self.boc_protection_pattern = "[NH2,NH1][C](=O)OC(C)(C)C"  # Boc-protected amine
        self.free_amine_pattern = "[NH2,NH1]"
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track Boc protection/deprotection events
        boc_events = []
        for i, rxn in enumerate(reactions):
            if self.is_boc_protection(rxn):
                boc_events.append(('protection', i))
            elif self.is_boc_deprotection(rxn):
                boc_events.append(('deprotection', i))
        
        # Check for cycling pattern: deprotection followed by re-protection
        cycling_detected = self.detect_cycling_pattern(boc_events)
        
        condition = cycling_detected == self.deprotection_reprotection
        return condition, len(reactions)
    
    def is_boc_protection(self, rxn):
        """Check if reaction involves Boc protection of an amine"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            # Check if reactants have free amine and products have Boc-protected amine
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if not all(reactant_mols + product_mols):
                return False
            
            # Look for free amine in reactants
            free_amine_in_reactants = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.free_amine_pattern))
                for mol in reactant_mols
            )
            
            # Look for Boc-protected amine in products
            boc_in_products = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.boc_protection_pattern))
                for mol in product_mols
            )
            
            # Also check for Boc reagents (tert-butyl dicarbonate, Boc2O)
            boc_reagent_pattern = "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C"  # Boc2O
            boc_reagent_present = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(boc_reagent_pattern))
                for mol in reactant_mols
            )
            
            return free_amine_in_reactants and boc_in_products and boc_reagent_present
            
        except:
            return False
    
    def is_boc_deprotection(self, rxn):
        """Check if reaction involves Boc deprotection to reveal amine"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if not all(reactant_mols + product_mols):
                return False
            
            # Look for Boc-protected amine in reactants
            boc_in_reactants = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.boc_protection_pattern))
                for mol in reactant_mols
            )
            
            # Look for free amine in products
            free_amine_in_products = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.free_amine_pattern))
                for mol in product_mols
            )
            
            # Check for typical deprotection conditions (TFA, HCl, etc.)
            acid_patterns = ["C(=O)(O)C(F)(F)F", "Cl"]  # TFA, HCl
            acid_present = any(
                any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for pattern in acid_patterns)
                for mol in reactant_mols
            )
            
            return boc_in_reactants and free_amine_in_products
            
        except:
            return False
    
    def detect_cycling_pattern(self, boc_events):
        """Detect if there's a deprotection followed by re-protection pattern"""
        if len(boc_events) < 2:
            return False
        
        # Look for deprotection followed by protection pattern
        for i in range(len(boc_events) - 1):
            current_event, current_step = boc_events[i]
            next_event, next_step = boc_events[i + 1]
            
            if current_event == 'deprotection' and next_event == 'protection':
                # Additional check: ensure they're not too far apart in the sequence
                if next_step - current_step <= 5:  # Within 5 reaction steps
                    return True
        
        return False
