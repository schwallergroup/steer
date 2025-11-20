"""Generated evaluation code for: Sequential protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates whether a synthesis route uses a sequential protecting group strategy
    with multiple orthogonal protecting groups (Boc, benzyl ester, TBDPS).
    """
    
    def __init__(self, config):
        self.protecting_groups = config.get("protecting_groups", ["Boc", "benzyl_ester", "TBDPS"])
        self.strategy = config.get("strategy", "sequential")
        self.min_groups_required = len(self.protecting_groups)
        
        # Define SMARTS patterns for each protecting group
        self.pg_patterns = {
            "Boc": "[NH1,NH0]-C(=O)OC(C)(C)C",  # tert-butoxycarbonyl
            "benzyl_ester": "C(=O)OCc1ccccc1",   # benzyl ester
            "TBDPS": "O[Si](c1ccccc1)(c2ccccc2)C(C)(C)C"  # tert-butyldiphenylsilyl
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        total_reactions = len(reactions)
        
        if total_reactions == 0:
            return False, 0
        
        # Track protecting group installations and removals
        pg_events = []
        
        for i, rxn in enumerate(reactions):
            depth = i + 1
            
            # Check for protecting group installation
            installed_pgs = self.detect_pg_installation(rxn)
            for pg in installed_pgs:
                pg_events.append(("install", pg, depth))
            
            # Check for protecting group removal
            removed_pgs = self.detect_pg_removal(rxn)
            for pg in removed_pgs:
                pg_events.append(("remove", pg, depth))
        
        # Evaluate sequential strategy
        condition_met = self.evaluate_sequential_strategy(pg_events)
        
        return condition_met, total_reactions
    
    def detect_pg_installation(self, rxn):
        """Detect protecting group installation in a reaction"""
        installed_pgs = []
        
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return installed_pgs
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
            
            if not all(reactants) or not all(products):
                return installed_pgs
            
            # Check each protecting group pattern
            for pg_name in self.protecting_groups:
                pattern = Chem.MolFromSmarts(self.pg_patterns[pg_name])
                if pattern is None:
                    continue
                
                # Count occurrences in reactants vs products
                reactant_matches = sum(len(mol.GetSubstructMatches(pattern)) for mol in reactants)
                product_matches = sum(len(mol.GetSubstructMatches(pattern)) for mol in products)
                
                # Installation: more occurrences in products than reactants
                if product_matches > reactant_matches:
                    installed_pgs.append(pg_name)
                    
        except Exception:
            pass
            
        return installed_pgs
    
    def detect_pg_removal(self, rxn):
        """Detect protecting group removal in a reaction"""
        removed_pgs = []
        
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return removed_pgs
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
            
            if not all(reactants) or not all(products):
                return removed_pgs
            
            # Check each protecting group pattern
            for pg_name in self.protecting_groups:
                pattern = Chem.MolFromSmarts(self.pg_patterns[pg_name])
                if pattern is None:
                    continue
                
                # Count occurrences in reactants vs products
                reactant_matches = sum(len(mol.GetSubstructMatches(pattern)) for mol in reactants)
                product_matches = sum(len(mol.GetSubstructMatches(pattern)) for mol in products)
                
                # Removal: fewer occurrences in products than reactants
                if reactant_matches > product_matches:
                    removed_pgs.append(pg_name)
                    
        except Exception:
            pass
            
        return removed_pgs
    
    def evaluate_sequential_strategy(self, pg_events):
        """Evaluate if the protecting group events follow a sequential strategy"""
        if len(pg_events) < self.min_groups_required:
            return False
        
        # Count unique protecting groups used
        used_pgs = set()
        for event_type, pg, depth in pg_events:
            if event_type == "install":
                used_pgs.add(pg)
        
        # Must use at least the minimum number of different protecting groups
        if len(used_pgs) < self.min_groups_required:
            return False
        
        # Check for sequential pattern: installations should occur before removals
        # and different protecting groups should be used orthogonally
        pg_lifecycles = {}
        
        for event_type, pg, depth in pg_events:
            if pg not in pg_lifecycles:
                pg_lifecycles[pg] = []
            pg_lifecycles[pg].append((event_type, depth))
        
        # Verify each protecting group has proper install->remove lifecycle
        valid_lifecycles = 0
        for pg in used_pgs:
            if pg in pg_lifecycles:
                events = sorted(pg_lifecycles[pg], key=lambda x: x[1])  # Sort by depth
                
                # Look for install followed by remove pattern
                for i in range(len(events) - 1):
                    if events[i][0] == "install" and events[i+1][0] == "remove":
                        valid_lifecycles += 1
                        break
        
        # Sequential strategy requires multiple protecting groups with proper lifecycles
        return valid_lifecycles >= 2 and len(used_pgs) >= 2
