"""Generated evaluation code for: Multiple protecting group strategy with Boc and benzyl"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates routes based on the use of multiple orthogonal protecting groups.
    Checks for the presence of specified protecting groups and their usage pattern.
    """
    
    def __init__(self, config):
        self.protecting_groups = config["protecting_groups"]
        self.required_count = config["count"]
        self.strategy = config["strategy"]
        
        # Define SMARTS patterns for common protecting groups
        self.pg_patterns = {
            "Boc": "[N:1][C](=O)OC(C)(C)C",  # tert-butoxycarbonyl
            "benzyl": "[O:1]Cc1ccccc1",  # benzyl ether
            "MOM": "[O:1]COC",  # methoxymethyl
            "Cbz": "[N:1][C](=O)OCc1ccccc1",  # carbobenzyloxy
            "TBDMS": "[O:1][Si](C)(C)C(C)(C)C",  # tert-butyldimethylsilyl
            "Ac": "[O:1][C](=O)C",  # acetyl
            "Ts": "[N:1]S(=O)(=O)c1ccc(C)cc1",  # tosyl
            "PMB": "[O:1]Cc1ccc(OC)cc1"  # para-methoxybenzyl
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group operations
        protection_reactions = []
        deprotection_reactions = []
        
        for rxn in reactions:
            pg_ops = self.analyze_protecting_group_operations(rxn)
            if pg_ops['protections']:
                protection_reactions.extend(pg_ops['protections'])
            if pg_ops['deprotections']:
                deprotection_reactions.extend(pg_ops['deprotections'])
        
        # Check if strategy requirements are met
        condition = self.evaluate_strategy(protection_reactions, deprotection_reactions)
        
        return condition, len(reactions)
    
    def analyze_protecting_group_operations(self, rxn):
        """Analyze a reaction for protecting group installation/removal"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return {'protections': [], 'deprotections': []}
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactants.split(".") if smi]
            product_mols = [Chem.MolFromSmiles(smi) for smi in products.split(".") if smi]
            
            if not all(reactant_mols) or not all(product_mols):
                return {'protections': [], 'deprotections': []}
                
        except:
            return {'protections': [], 'deprotections': []}
        
        protections = []
        deprotections = []
        
        for pg_name in self.protecting_groups:
            if pg_name in self.pg_patterns:
                pattern = Chem.MolFromSmarts(self.pg_patterns[pg_name])
                if pattern is None:
                    continue
                    
                # Count PG in reactants vs products
                reactant_matches = sum(len(mol.GetSubstructMatches(pattern)) 
                                     for mol in reactant_mols)
                product_matches = sum(len(mol.GetSubstructMatches(pattern)) 
                                    for mol in product_mols)
                
                if product_matches > reactant_matches:
                    # Protection reaction
                    protections.append(pg_name)
                elif reactant_matches > product_matches:
                    # Deprotection reaction
                    deprotections.append(pg_name)
        
        return {'protections': protections, 'deprotections': deprotections}
    
    def evaluate_strategy(self, protections, deprotections):
        """Evaluate if the protecting group strategy meets requirements"""
        unique_protections = set(protections)
        unique_deprotections = set(deprotections)
        
        if self.strategy == "orthogonal":
            # For orthogonal strategy, check if we have the required number
            # of different protecting groups used
            total_unique_pgs = unique_protections | unique_deprotections
            
            # Must have at least the required count of different PGs
            if len(total_unique_pgs) < self.required_count:
                return False
                
            # Check that the specified protecting groups are present
            specified_pgs = set(self.protecting_groups)
            if not specified_pgs.issubset(total_unique_pgs):
                return False
                
            # For orthogonal strategy, ideally each protection should have
            # a corresponding deprotection
            for pg in unique_protections:
                if pg not in unique_deprotections:
                    # This is acceptable - some PGs might remain at the end
                    pass
                    
            return True
            
        elif self.strategy == "sequential":
            # Sequential strategy requires specific order
            return len(unique_protections) >= self.required_count
            
        return False
    
    def route_scoring(self, x) -> float:
        """Convert condition result to score"""
        if x < 0:
            return 0  # Strategy not found
        else:
            # Earlier implementation of strategy is better
            return max(0, 1 - x)
