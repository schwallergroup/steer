"""Generated evaluation code for: Sequential protecting group cycling on same functional group"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupCycling(MultiRxnCondBase):
    """
    Detects sequential protecting group cycling on the same functional group.
    Identifies routes that deprotect a functional group and then re-protect
    the same group, creating a pointless cycle.
    """
    
    def __init__(self, config):
        self.functional_group = config["functional_group"]
        self.protection_count = config["protection_count"]
        self.sequential = config["sequential"]
        self.same_group = config["same_group"]
        
        # Define protecting group patterns for different functional groups
        self.protection_patterns = {
            "ketone": {
                "acetal": "[CH]([OR])([OR])",
                "ketal": "[C]([OR])([OR])",
                "silyl_enol": "[C]=[C][OSi]"
            },
            "alcohol": {
                "silyl": "[OH0][Si]",
                "benzyl": "[OH0]Cc1ccccc1",
                "acetyl": "[OH0]C(=O)C"
            },
            "amine": {
                "boc": "[NH0]C(=O)OC(C)(C)C",
                "cbz": "[NH0]C(=O)OCc1ccccc1",
                "acetyl": "[NH0]C(=O)C"
            }
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        if len(reactions) < self.protection_count:
            return False, len(reactions)
        
        # Track functional group states through the route
        fg_states = self._track_functional_group_states(reactions)
        
        # Look for cycling pattern: protected -> deprotected -> protected
        cycling_detected = self._detect_cycling_pattern(fg_states)
        
        return cycling_detected, len(reactions)
    
    def _track_functional_group_states(self, reactions):
        """Track the protection state of functional groups through reactions"""
        fg_states = []
        
        for rxn in reactions:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                continue
                
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                continue
            
            # Check if this reaction involves protection/deprotection
            is_protection = self._is_protection_reaction(product, reactants)
            is_deprotection = self._is_deprotection_reaction(product, reactants)
            
            if is_protection:
                fg_states.append("protected")
            elif is_deprotection:
                fg_states.append("deprotected")
            else:
                fg_states.append("unchanged")
        
        return fg_states
    
    def _is_protection_reaction(self, product, reactants):
        """Check if reaction protects the target functional group"""
        if self.functional_group not in self.protection_patterns:
            return False
        
        patterns = self.protection_patterns[self.functional_group]
        
        # Check if product has more protected groups than reactants
        product_protected = sum(self._count_protected_groups(product, patterns))
        reactant_protected = sum(sum(self._count_protected_groups(r, patterns)) for r in reactants)
        
        return product_protected > reactant_protected
    
    def _is_deprotection_reaction(self, product, reactants):
        """Check if reaction deprotects the target functional group"""
        if self.functional_group not in self.protection_patterns:
            return False
        
        patterns = self.protection_patterns[self.functional_group]
        
        # Check if product has fewer protected groups than reactants
        product_protected = sum(self._count_protected_groups(product, patterns))
        reactant_protected = sum(sum(self._count_protected_groups(r, patterns)) for r in reactants)
        
        return product_protected < reactant_protected
    
    def _count_protected_groups(self, mol, patterns):
        """Count protected functional groups in molecule"""
        if not mol:
            return [0] * len(patterns)
        
        counts = []
        for pattern_name, smarts in patterns.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                matches = mol.GetSubstructMatches(pattern)
                counts.append(len(matches))
            else:
                counts.append(0)
        
        return counts
    
    def _detect_cycling_pattern(self, fg_states):
        """Detect if there's a cycling pattern in functional group states"""
        if len(fg_states) < self.protection_count:
            return False
        
        # Look for pattern: protected -> deprotected -> protected (or vice versa)
        cycling_count = 0
        
        for i in range(len(fg_states) - 1):
            current_state = fg_states[i]
            next_state = fg_states[i + 1]
            
            # Check for state changes indicating cycling
            if (current_state == "protected" and next_state == "deprotected") or \
               (current_state == "deprotected" and next_state == "protected"):
                cycling_count += 1
        
        # Require at least the specified number of protection/deprotection cycles
        return cycling_count >= self.protection_count
