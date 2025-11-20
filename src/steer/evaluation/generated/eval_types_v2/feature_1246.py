"""Generated evaluation code for: Extensive protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes for extensive protecting group cycling strategies.
    Checks for multiple protection-deprotection cycles using specified protecting groups
    on the same functional groups.
    """
    
    def __init__(self, config):
        self.min_cycles = config["protection_deprotection_cycles"]
        self.protecting_groups = config["protecting_groups"]
        self.same_functionality = config.get("same_functionality", True)
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "benzylidene": "[CH]1O[CH2][CH][CH2]O1",  # Benzylidene acetal
            "acetate": "[CH2,CH,CH3]OC(=O)[CH3]",     # Acetate ester
            "benzyl": "[CH2,CH,CH3]O[CH2]c1ccccc1"    # Benzyl ether
        }
        
        # Compile patterns
        self.compiled_patterns = {}
        for pg in self.protecting_groups:
            if pg in self.pg_patterns:
                self.compiled_patterns[pg] = Chem.MolFromSmarts(self.pg_patterns[pg])

    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if the route contains the required protecting group cycling strategy."""
        reactions = self.get_rxns(d)
        
        # Track protection/deprotection events
        pg_events = []
        
        for i, rxn in enumerate(reactions):
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                continue
                
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Check for protection (PG appears in product)
            for pg_name, pattern in self.compiled_patterns.items():
                if pattern is None:
                    continue
                    
                reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
                product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
                
                reactant_mols = [mol for mol in reactant_mols if mol is not None]
                product_mols = [mol for mol in product_mols if mol is not None]
                
                # Count PG occurrences
                reactant_pg_count = sum(len(mol.GetSubstructMatches(pattern)) for mol in reactant_mols)
                product_pg_count = sum(len(mol.GetSubstructMatches(pattern)) for mol in product_mols)
                
                if product_pg_count > reactant_pg_count:
                    # Protection event
                    pg_events.append({
                        'type': 'protection',
                        'pg': pg_name,
                        'step': i,
                        'count_change': product_pg_count - reactant_pg_count
                    })
                elif reactant_pg_count > product_pg_count:
                    # Deprotection event
                    pg_events.append({
                        'type': 'deprotection',
                        'pg': pg_name,
                        'step': i,
                        'count_change': reactant_pg_count - product_pg_count
                    })
        
        # Count complete cycles
        cycles = self._count_cycles(pg_events)
        
        condition_met = cycles >= self.min_cycles
        return condition_met, len(reactions)
    
    def _count_cycles(self, pg_events):
        """Count complete protection-deprotection cycles."""
        cycles = 0
        
        if self.same_functionality:
            # Track cycles for the same protecting group type
            for pg_name in self.protecting_groups:
                pg_specific_events = [e for e in pg_events if e['pg'] == pg_name]
                cycles += self._count_cycles_for_pg(pg_specific_events)
        else:
            # Count cycles across different protecting groups
            cycles = self._count_cycles_for_pg(pg_events)
        
        return cycles
    
    def _count_cycles_for_pg(self, events):
        """Count cycles for a specific set of events."""
        if not events:
            return 0
            
        # Sort events by step
        events.sort(key=lambda x: x['step'])
        
        cycles = 0
        protection_count = 0
        
        for event in events:
            if event['type'] == 'protection':
                protection_count += event['count_change']
            elif event['type'] == 'deprotection':
                # Each deprotection can complete a cycle if there was a prior protection
                deprotection_count = event['count_change']
                completed_cycles = min(protection_count, deprotection_count)
                cycles += completed_cycles
                protection_count -= completed_cycles
        
        return cycles
    
    def route_scoring(self, cycles_found):
        """Convert cycle count to score (0-10)."""
        if cycles_found < 0:
            return 0
        
        if cycles_found >= self.min_cycles:
            # Bonus for exceeding minimum cycles
            return min(10, 7 + (cycles_found - self.min_cycles))
        else:
            # Partial credit for some cycles
            return (cycles_found / self.min_cycles) * 7
