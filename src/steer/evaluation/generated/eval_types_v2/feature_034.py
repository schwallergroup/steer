"""Generated evaluation code for: Extensive protecting group cycling on phenols"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ExtensiveProtectingGroupCycling(MultiRxnCondBase):
    """
    Checks for extensive protecting group cycling on phenols.
    Counts protection/deprotection cycles for specified protecting groups on phenolic compounds.
    """
    
    def __init__(self, config):
        self.protecting_groups = config["protecting_groups"]
        self.functional_group = config["functional_group"]
        self.cycle_count = config["cycle_count"]
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "benzyl": "[OH1][c]",  # phenol
            "acetyl": "[OH0]([C](=O)[CH3])[c]",  # acetyl protected phenol
            "THP": "[OH0]([CH1]1[OH0][CH2][CH2][CH2][CH2]1)[c]"  # THP protected phenol
        }
        
        # Phenol pattern
        self.phenol_pattern = "[OH1][c]"
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protection/deprotection events
        protection_events = []
        
        for rxn in reactions:
            event = self.detect_protection_deprotection(rxn)
            if event:
                protection_events.append(event)
        
        # Count cycles for each protecting group
        cycles = self.count_cycles(protection_events)
        
        # Check if any protecting group has >= cycle_count cycles
        condition_met = any(count >= self.cycle_count for count in cycles.values())
        
        return condition_met, len(reactions)
    
    def detect_protection_deprotection(self, rxn):
        """Detect if a reaction is protection or deprotection of phenol"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return None
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return None
            
            # Check for protection (phenol -> protected phenol)
            for pg_name in self.protecting_groups:
                if self.is_protection_reaction(reactant_mols, product_mols, pg_name):
                    return {"type": "protection", "group": pg_name}
                    
            # Check for deprotection (protected phenol -> phenol)
            for pg_name in self.protecting_groups:
                if self.is_deprotection_reaction(reactant_mols, product_mols, pg_name):
                    return {"type": "deprotection", "group": pg_name}
                    
        except Exception:
            pass
            
        return None
    
    def is_protection_reaction(self, reactants, products, pg_name):
        """Check if reaction is protection of phenol with specific protecting group"""
        # Look for phenol in reactants
        has_phenol_reactant = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.phenol_pattern)) 
            for mol in reactants
        )
        
        # Look for protected phenol in products
        if pg_name in self.pg_patterns:
            has_protected_product = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.pg_patterns[pg_name])) 
                for mol in products
            )
            
            return has_phenol_reactant and has_protected_product
            
        return False
    
    def is_deprotection_reaction(self, reactants, products, pg_name):
        """Check if reaction is deprotection of specific protecting group to give phenol"""
        # Look for protected phenol in reactants
        if pg_name in self.pg_patterns:
            has_protected_reactant = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.pg_patterns[pg_name])) 
                for mol in reactants
            )
        else:
            has_protected_reactant = False
            
        # Look for phenol in products
        has_phenol_product = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.phenol_pattern)) 
            for mol in products
        )
        
        return has_protected_reactant and has_phenol_product
    
    def count_cycles(self, events):
        """Count complete protection-deprotection cycles for each protecting group"""
        cycles = {pg: 0 for pg in self.protecting_groups}
        
        # Track state for each protecting group
        protection_count = {pg: 0 for pg in self.protecting_groups}
        
        for event in events:
            pg = event["group"]
            if event["type"] == "protection":
                protection_count[pg] += 1
            elif event["type"] == "deprotection":
                if protection_count[pg] > 0:
                    cycles[pg] += 1
                    protection_count[pg] -= 1
                    
        return cycles
