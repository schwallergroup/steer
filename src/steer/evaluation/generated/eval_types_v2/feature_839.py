"""Generated evaluation code for: Sequential orthogonal protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialOrthogonalProtectingGroup(MultiRxnCondBase):
    """
    Evaluates routes that use sequential orthogonal protecting group strategies.
    Checks for the presence of specified protecting groups applied to target
    functional groups in an orthogonal manner.
    """
    
    def __init__(self, config):
        self.protecting_groups = config.get("protecting_groups", [])
        self.strategy_type = config.get("strategy_type", "sequential_orthogonal")
        self.functional_groups_protected = config.get("functional_groups_protected", [])
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "TBDMS": "[Si](C)(C)C(C)(C)C",  # tert-butyldimethylsilyl
            "trityl": "C(c1ccccc1)(c2ccccc2)(c3ccccc3)",  # trityl group
            "acetyl": "C(=O)C",  # acetyl group
            "Boc": "C(=O)OC(C)(C)C",  # tert-butoxycarbonyl
            "Cbz": "C(=O)OCc1ccccc1",  # carbobenzyloxy
            "TMS": "[Si](C)(C)C"  # trimethylsilyl
        }
        
        # Define SMARTS patterns for protected functional groups
        self.protected_fg_patterns = {
            "secondary_alcohol": "[CH]O[#6]",  # secondary alcohol protected
            "primary_alcohol": "[CH2]O[#6]",  # primary alcohol protected
            "amine": "N[#6]",  # protected amine
            "carboxylic_acid": "C(=O)O[#6]"  # protected carboxylic acid
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if the route uses the specified orthogonal protecting group strategy"""
        reactions = self.get_rxns(d)
        
        # Track protecting group installations and removals
        pg_installations = {pg: [] for pg in self.protecting_groups}
        pg_removals = {pg: [] for pg in self.protecting_groups}
        protected_groups_found = {fg: [] for fg in self.functional_groups_protected}
        
        for i, rxn in enumerate(reactions):
            # Check for protecting group installations
            for pg in self.protecting_groups:
                if self.detect_pg_installation(rxn, pg):
                    pg_installations[pg].append(i)
            
            # Check for protecting group removals
            for pg in self.protecting_groups:
                if self.detect_pg_removal(rxn, pg):
                    pg_removals[pg].append(i)
            
            # Check for protected functional groups
            for fg in self.functional_groups_protected:
                if self.detect_protected_functional_group(rxn, fg):
                    protected_groups_found[fg].append(i)
        
        # Evaluate orthogonal strategy
        condition = self.evaluate_orthogonal_strategy(
            pg_installations, pg_removals, protected_groups_found
        )
        
        return condition, len(reactions)
    
    def detect_pg_installation(self, rxn, protecting_group):
        """Detect installation of a specific protecting group"""
        if protecting_group not in self.pg_patterns:
            return False
        
        pattern = Chem.MolFromSmarts(self.pg_patterns[protecting_group])
        if pattern is None:
            return False
        
        # Check if protecting group appears in products but not in reactants
        reactants_have_pg = any(
            mol.HasSubstructMatch(pattern) 
            for mol in rxn['reactants'] if mol is not None
        )
        products_have_pg = any(
            mol.HasSubstructMatch(pattern) 
            for mol in rxn['products'] if mol is not None
        )
        
        return products_have_pg and not reactants_have_pg
    
    def detect_pg_removal(self, rxn, protecting_group):
        """Detect removal of a specific protecting group"""
        if protecting_group not in self.pg_patterns:
            return False
        
        pattern = Chem.MolFromSmarts(self.pg_patterns[protecting_group])
        if pattern is None:
            return False
        
        # Check if protecting group appears in reactants but not in products
        reactants_have_pg = any(
            mol.HasSubstructMatch(pattern) 
            for mol in rxn['reactants'] if mol is not None
        )
        products_have_pg = any(
            mol.HasSubstructMatch(pattern) 
            for mol in rxn['products'] if mol is not None
        )
        
        return reactants_have_pg and not products_have_pg
    
    def detect_protected_functional_group(self, rxn, functional_group):
        """Detect protection of a specific functional group"""
        if functional_group not in self.protected_fg_patterns:
            return False
        
        pattern = Chem.MolFromSmarts(self.protected_fg_patterns[functional_group])
        if pattern is None:
            return False
        
        # Check if protected functional group appears in products
        return any(
            mol.HasSubstructMatch(pattern) 
            for mol in rxn['products'] if mol is not None
        )
    
    def evaluate_orthogonal_strategy(self, pg_installations, pg_removals, protected_groups_found):
        """Evaluate if the protecting group strategy is truly orthogonal and sequential"""
        
        # Check that required protecting groups are used
        required_pgs_used = sum(1 for pg in self.protecting_groups 
                               if len(pg_installations[pg]) > 0)
        
        if required_pgs_used < len(self.protecting_groups):
            return False
        
        # Check that target functional groups are protected
        required_fgs_protected = sum(1 for fg in self.functional_groups_protected 
                                   if len(protected_groups_found[fg]) > 0)
        
        if required_fgs_protected < len(self.functional_groups_protected):
            return False
        
        # Check for orthogonal strategy: protecting groups should be installed
        # and removed at different stages (non-overlapping)
        all_installation_steps = []
        all_removal_steps = []
        
        for pg in self.protecting_groups:
            all_installation_steps.extend(pg_installations[pg])
            all_removal_steps.extend(pg_removals[pg])
        
        # For sequential orthogonal strategy, we expect:
        # 1. Multiple different protecting groups used
        # 2. Protecting groups installed and removed at different times
        # 3. No simultaneous removal of different protecting groups (orthogonal)
        
        if self.strategy_type == "sequential_orthogonal":
            # Check that removals happen at different steps for different PGs
            removal_steps_by_pg = {}
            for pg in self.protecting_groups:
                if pg_removals[pg]:
                    removal_steps_by_pg[pg] = set(pg_removals[pg])
            
            # Ensure orthogonality: different PGs removed at different steps
            if len(removal_steps_by_pg) > 1:
                all_removal_sets = list(removal_steps_by_pg.values())
                for i in range(len(all_removal_sets)):
                    for j in range(i + 1, len(all_removal_sets)):
                        if all_removal_sets[i].intersection(all_removal_sets[j]):
                            return False  # Non-orthogonal removal detected
        
        return True
