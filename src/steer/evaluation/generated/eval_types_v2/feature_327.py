"""Generated evaluation code for: Multiple protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes based on protecting group cycling strategy.
    Checks for multiple protection/deprotection cycles of specified functional groups
    with sequential swaps between different protecting groups.
    """
    
    def __init__(self, config):
        self.target_cycles = config["parameters"]["protection_deprotection_cycles"]
        self.functional_group = config["parameters"]["functional_group"]
        self.require_sequential = config["parameters"]["sequential_swaps"]
        
        # Define SMARTS patterns for functional groups and protecting groups
        self.fg_patterns = {
            "phenol": "[OH1][c]",
            "amine": "[NH2,NH1]",
            "carboxylic_acid": "[CX3](=O)[OH1]",
            "alcohol": "[OH1][CX4]"
        }
        
        self.protecting_patterns = {
            "MOM": "[CH2]O[CH3]",  # Methoxymethyl
            "TIPS": "[Si]([CH3])([CH3])[CH](C)C",  # Triisopropylsilyl
            "acetate": "C(=O)[CH3]",  # Acetyl
            "Boc": "C(=O)OC(C)(C)C",  # tert-Butoxycarbonyl
            "benzyl": "[CH2]c1ccccc1",  # Benzyl
            "TBS": "[Si](C)(C)C(C)(C)C"  # tert-Butyldimethylsilyl
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protection/deprotection events
        pg_events = []
        
        for i, rxn in enumerate(reactions):
            event_type, pg_type = self.analyze_protection_event(rxn)
            if event_type:
                pg_events.append({
                    'depth': i,
                    'type': event_type,
                    'protecting_group': pg_type,
                    'functional_group': self.functional_group
                })
        
        # Count cycles and check for sequential swaps
        cycles = self.count_protection_cycles(pg_events)
        has_sequential = self.check_sequential_swaps(pg_events) if self.require_sequential else True
        
        condition_met = cycles >= self.target_cycles and has_sequential
        
        return condition_met, len(reactions)
    
    def analyze_protection_event(self, rxn):
        """
        Analyze a reaction to determine if it's a protection or deprotection event.
        Returns (event_type, protecting_group_type) where event_type is 'protect'/'deprotect'/None
        """
        rxn_parts = rxn.split(">>")
        reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
        
        if not all(reactants + products):
            return None, None
        
        fg_pattern = self.fg_patterns.get(self.functional_group)
        if not fg_pattern:
            return None, None
        
        fg_mol = Chem.MolFromSmarts(fg_pattern)
        
        # Check reactants and products for functional group and protecting groups
        reactant_has_fg = any(mol.HasSubstructMatch(fg_mol) for mol in reactants if mol)
        product_has_fg = any(mol.HasSubstructMatch(fg_mol) for mol in products if mol)
        
        # Detect protecting groups in reactants and products
        reactant_pgs = set()
        product_pgs = set()
        
        for pg_name, pg_pattern in self.protecting_patterns.items():
            pg_mol = Chem.MolFromSmarts(pg_pattern)
            if any(mol.HasSubstructMatch(pg_mol) for mol in reactants if mol):
                reactant_pgs.add(pg_name)
            if any(mol.HasSubstructMatch(pg_mol) for mol in products if mol):
                product_pgs.add(pg_name)
        
        # Determine if this is protection or deprotection
        if reactant_has_fg and not product_has_fg and product_pgs:
            # Protection: free FG -> protected FG
            return 'protect', list(product_pgs)[0]
        elif not reactant_has_fg and product_has_fg and reactant_pgs:
            # Deprotection: protected FG -> free FG
            return 'deprotect', list(reactant_pgs)[0]
        elif reactant_pgs and product_pgs and reactant_pgs != product_pgs:
            # Protection swap: one PG -> different PG
            new_pg = product_pgs - reactant_pgs
            if new_pg:
                return 'protect', list(new_pg)[0]
        
        return None, None
    
    def count_protection_cycles(self, pg_events):
        """
        Count complete protection-deprotection cycles for the target functional group.
        """
        cycles = 0
        protection_stack = []
        
        for event in pg_events:
            if event['type'] == 'protect':
                protection_stack.append(event['protecting_group'])
            elif event['type'] == 'deprotect' and protection_stack:
                protection_stack.pop()
                cycles += 1
        
        return cycles
    
    def check_sequential_swaps(self, pg_events):
        """
        Check if there are sequential swaps between different protecting groups.
        """
        if len(pg_events) < 2:
            return False
        
        protecting_groups_used = set()
        has_swaps = False
        
        for event in pg_events:
            if event['type'] == 'protect':
                if event['protecting_group'] in protecting_groups_used:
                    has_swaps = True
                protecting_groups_used.add(event['protecting_group'])
        
        return len(protecting_groups_used) >= 2 and has_swaps
    
    def route_scoring(self, condition_met, total_depth):
        """
        Score the route based on whether the protecting group strategy is present.
        """
        if condition_met:
            return 10.0  # Perfect score if strategy is found
        else:
            return 0.0   # No score if strategy is not present
