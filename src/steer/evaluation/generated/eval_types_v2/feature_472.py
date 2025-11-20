"""Generated evaluation code for: Sequential protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes based on sequential protecting group cycling strategy.
    Looks for patterns where protecting groups are removed, reactions performed,
    and then re-protected with different groups in a cycling manner.
    """
    
    def __init__(self, config):
        self.strategy_type = config.get("strategy_type", "sequential_cycling")
        self.min_protection_steps = config.get("protection_steps", 2)
        self.min_deprotection_steps = config.get("deprotection_steps", 1)
        
        # Common protecting group patterns
        self.protecting_groups = {
            "boc": "[CH3][CH3][CH3]OC(=O)",  # tert-butoxycarbonyl
            "cbz": "c1ccccc1[CH2]OC(=O)",    # benzyloxycarbonyl
            "fmoc": "c1ccc2c(c1)cc3ccccc3c2[CH2][CH2]OC(=O)",  # fluorenylmethoxycarbonyl
            "benzyl": "c1ccccc1[CH2]",       # benzyl
            "acetyl": "[CH3]C(=O)",          # acetyl
            "tosyl": "c1ccc(cc1)[S](=O)(=O)", # tosyl
            "trityl": "c1ccccc1C(c2ccccc2)(c3ccccc3)" # trityl
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        protection_events = []
        deprotection_events = []
        
        for i, rxn in enumerate(reactions):
            protection_type = self.detect_protection_event(rxn)
            if protection_type:
                if protection_type == "protection":
                    protection_events.append(i)
                elif protection_type == "deprotection":
                    deprotection_events.append(i)
        
        # Check for sequential cycling pattern
        cycling_pattern = self.analyze_cycling_pattern(
            reactions, protection_events, deprotection_events
        )
        
        condition_met = (
            len(protection_events) >= self.min_protection_steps and
            len(deprotection_events) >= self.min_deprotection_steps and
            cycling_pattern
        )
        
        return condition_met, len(reactions)
    
    def detect_protection_event(self, rxn):
        """Detect if a reaction involves protection or deprotection"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return None
                
            reactants = Chem.MolFromSmiles(rxn_parts[0])
            products = Chem.MolFromSmiles(rxn_parts[1])
            
            if not reactants or not products:
                return None
            
            # Count protecting groups in reactants vs products
            reactant_pg_count = self.count_protecting_groups(reactants)
            product_pg_count = self.count_protecting_groups(products)
            
            if product_pg_count > reactant_pg_count:
                return "protection"
            elif reactant_pg_count > product_pg_count:
                return "deprotection"
            else:
                # Check for protecting group exchange (same count, different types)
                reactant_pg_types = self.identify_protecting_group_types(reactants)
                product_pg_types = self.identify_protecting_group_types(products)
                
                if reactant_pg_types != product_pg_types and len(reactant_pg_types) > 0:
                    return "protection"  # Treat exchange as protection event
                    
        except Exception:
            pass
        
        return None
    
    def count_protecting_groups(self, mol):
        """Count total number of protecting groups in molecule"""
        count = 0
        for pg_name, pattern in self.protecting_groups.items():
            try:
                pg_mol = Chem.MolFromSmarts(pattern)
                if pg_mol and mol.HasSubstructMatch(pg_mol):
                    matches = mol.GetSubstructMatches(pg_mol)
                    count += len(matches)
            except Exception:
                continue
        return count
    
    def identify_protecting_group_types(self, mol):
        """Identify types of protecting groups present"""
        pg_types = set()
        for pg_name, pattern in self.protecting_groups.items():
            try:
                pg_mol = Chem.MolFromSmarts(pattern)
                if pg_mol and mol.HasSubstructMatch(pg_mol):
                    pg_types.add(pg_name)
            except Exception:
                continue
        return pg_types
    
    def analyze_cycling_pattern(self, reactions, protection_events, deprotection_events):
        """Analyze if protection/deprotection events form a cycling pattern"""
        if len(protection_events) < 2 or len(deprotection_events) < 1:
            return False
        
        # Check for alternating or sequential pattern
        all_events = []
        for p_idx in protection_events:
            all_events.append((p_idx, "protection"))
        for d_idx in deprotection_events:
            all_events.append((d_idx, "deprotection"))
        
        # Sort by reaction index
        all_events.sort(key=lambda x: x[0])
        
        # Look for at least one deprotection followed by protection
        found_cycle = False
        for i in range(len(all_events) - 1):
            current_event = all_events[i][1]
            next_event = all_events[i + 1][1]
            
            if current_event == "deprotection" and next_event == "protection":
                found_cycle = True
                break
        
        return found_cycle
