"""Generated evaluation code for: Multiple ester protecting group cycling steps"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EsterProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes that involve multiple ester protecting group cycling steps.
    Checks for sequential conversion between different ester types (methyl, ethyl, tert-butyl)
    through hydrolysis-esterification cycles.
    """
    
    def __init__(self, config):
        self.cycle_count = config.get("cycle_count", 3)
        self.ester_types = config.get("ester_types", ["methyl", "ethyl", "tert-butyl"])
        
        # Define SMARTS patterns for different ester types
        self.ester_patterns = {
            "methyl": "[C:1](=O)[O:2][CH3:3]",
            "ethyl": "[C:1](=O)[O:2][CH2:3][CH3:4]", 
            "tert-butyl": "[C:1](=O)[O:2][C:3]([CH3])([CH3])[CH3]"
        }
        
        # Patterns for detecting hydrolysis (ester to carboxylic acid)
        self.carboxylic_acid_pattern = "[C:1](=O)[OH:2]"
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track ester transformations through the route
        ester_changes = []
        
        for rxn in reactions:
            change_type = self.detect_ester_transformation(rxn)
            if change_type:
                ester_changes.append(change_type)
        
        # Check if we have the required number of cycling steps
        cycle_count = self.count_protection_cycles(ester_changes)
        
        condition = cycle_count >= self.cycle_count
        return condition, len(reactions)
    
    def detect_ester_transformation(self, rxn):
        """Detect if a reaction involves ester protection/deprotection."""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return None
                
            reactants = Chem.MolFromSmiles(rxn_parts[0])
            products = Chem.MolFromSmiles(rxn_parts[1])
            
            if not reactants or not products:
                return None
            
            # Check for ester type in reactants and products
            reactant_ester = self.identify_ester_type(reactants)
            product_ester = self.identify_ester_type(products)
            
            # Check for carboxylic acid intermediate (hydrolysis step)
            reactant_acid = self.has_carboxylic_acid(reactants)
            product_acid = self.has_carboxylic_acid(products)
            
            # Classify transformation type
            if reactant_ester and product_acid:
                return f"deprotection_{reactant_ester}"
            elif reactant_acid and product_ester:
                return f"protection_{product_ester}"
            elif reactant_ester and product_ester and reactant_ester != product_ester:
                return f"exchange_{reactant_ester}_to_{product_ester}"
            
            return None
            
        except Exception:
            return None
    
    def identify_ester_type(self, mol):
        """Identify which type of ester is present in the molecule."""
        for ester_type, pattern in self.ester_patterns.items():
            if ester_type in self.ester_types:
                smarts = Chem.MolFromSmarts(pattern)
                if smarts and mol.HasSubstructMatch(smarts):
                    return ester_type
        return None
    
    def has_carboxylic_acid(self, mol):
        """Check if molecule contains a carboxylic acid group."""
        acid_smarts = Chem.MolFromSmarts(self.carboxylic_acid_pattern)
        return acid_smarts and mol.HasSubstructMatch(acid_smarts)
    
    def count_protection_cycles(self, ester_changes):
        """Count the number of complete protection/deprotection cycles."""
        cycles = 0
        i = 0
        
        while i < len(ester_changes) - 1:
            current_change = ester_changes[i]
            
            # Look for deprotection followed by protection (or direct exchange)
            if current_change.startswith("deprotection_"):
                # Find next protection step
                for j in range(i + 1, len(ester_changes)):
                    next_change = ester_changes[j]
                    if next_change.startswith("protection_"):
                        cycles += 1
                        i = j
                        break
                else:
                    i += 1
            elif current_change.startswith("exchange_"):
                cycles += 1
                i += 1
            else:
                i += 1
        
        # Also count sequences where we see multiple different ester types
        unique_esters_seen = set()
        for change in ester_changes:
            for ester_type in self.ester_types:
                if ester_type in change:
                    unique_esters_seen.add(ester_type)
        
        # If we see all expected ester types, that's indicative of cycling
        if len(unique_esters_seen) >= len(self.ester_types):
            cycles = max(cycles, len(unique_esters_seen) - 1)
        
        return cycles
