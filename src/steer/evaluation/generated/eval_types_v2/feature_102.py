"""Generated evaluation code for: Multiple protecting group cycling throughout synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates synthesis routes for multiple protecting group cycling patterns.
    Checks if the route contains the specified number of protection-deprotection
    cycles with protecting group swaps.
    """
    
    def __init__(self, config):
        self.protection_cycles = config["protection_cycles"]
        self.cycle_details = config["cycle_details"]
        
        # Define SMARTS patterns for common protecting groups
        self.protecting_group_patterns = {
            "Cbz": "[NH1][C](=O)[O][CH2]c1ccccc1",  # Benzyloxycarbonyl
            "Boc": "[NH1][C](=O)[O][C](C)(C)C",     # tert-Butyloxycarbonyl
            "trifluoroacetamide": "[NH1][C](=O)[C](F)(F)F",  # Trifluoroacetamide
            "acetamide": "[NH1][C](=O)[C]"          # Acetamide
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group events throughout the route
        protection_events = []
        
        for i, rxn in enumerate(reactions):
            event_type, pg_type = self.classify_protection_event(rxn)
            if event_type:
                protection_events.append({
                    'step': i,
                    'type': event_type,  # 'protect' or 'deprotect'
                    'group': pg_type
                })
        
        # Check if we have the required number of complete cycles
        cycles_found = self.count_protection_cycles(protection_events)
        condition_met = cycles_found >= self.protection_cycles
        
        return condition_met, len(reactions)
    
    def classify_protection_event(self, rxn):
        """
        Classify a reaction as protection or deprotection event.
        Returns (event_type, protecting_group_type) or (None, None)
        """
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return None, None
            
        reactants = Chem.MolFromSmiles(rxn_parts[0])
        products = Chem.MolFromSmiles(rxn_parts[1])
        
        if not reactants or not products:
            return None, None
        
        for pg_name, pattern in self.protecting_group_patterns.items():
            pg_mol = Chem.MolFromSmarts(pattern)
            if not pg_mol:
                continue
                
            reactant_has_pg = reactants.HasSubstructMatch(pg_mol)
            product_has_pg = products.HasSubstructMatch(pg_mol)
            
            if not reactant_has_pg and product_has_pg:
                return 'protect', pg_name
            elif reactant_has_pg and not product_has_pg:
                return 'deprotect', pg_name
        
        return None, None
    
    def count_protection_cycles(self, events):
        """
        Count complete protection cycles based on the cycle details.
        A cycle is: protect -> deprotect -> protect_again (with group swap)
        """
        cycles_completed = 0
        
        for cycle_detail in self.cycle_details:
            protect_group = cycle_detail["protect"]
            deprotect_group = cycle_detail["deprotect"]
            protect_again_group = cycle_detail["protect_again"]
            
            # Find instances of this cycle pattern
            i = 0
            while i < len(events) - 2:
                # Look for protection with the first group
                if (events[i]['type'] == 'protect' and 
                    events[i]['group'] == protect_group):
                    
                    # Look for subsequent deprotection of the same group
                    j = i + 1
                    while j < len(events):
                        if (events[j]['type'] == 'deprotect' and 
                            events[j]['group'] == deprotect_group):
                            
                            # Look for protection with different group
                            k = j + 1
                            while k < len(events):
                                if (events[k]['type'] == 'protect' and 
                                    events[k]['group'] == protect_again_group):
                                    cycles_completed += 1
                                    i = k  # Move past this cycle
                                    break
                                k += 1
                            break
                        j += 1
                    else:
                        i += 1
                else:
                    i += 1
        
        return cycles_completed
    
    def route_scoring(self, condition_met):
        """
        Score based on whether the required protection cycling pattern was found.
        Returns higher score if condition is met.
        """
        if condition_met:
            return 1.0  # Perfect score if cycles are present
        else:
            return 0.0  # No score if cycles are missing
