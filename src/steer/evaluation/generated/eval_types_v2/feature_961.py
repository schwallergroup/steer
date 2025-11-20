"""Generated evaluation code for: Sequential protecting group strategy for amine selectivity"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates whether a synthesis route uses sequential protecting group strategies
    for controlling amine reactivity through strategic Boc protection/deprotection.
    """
    
    def __init__(self, config):
        self.protecting_groups = config["protecting_groups"]
        self.strategy_type = config["strategy_type"]
        self.target_functionality = config["target_functionality"]
        
        # Define SMARTS patterns for Boc groups and related transformations
        self.boc_patterns = {
            "boc_protected_amine": "[NX3]C(=O)OC(C)(C)C",  # N-Boc protected amine
            "boc_anhydride": "C(C)(C)COC(=O)OC(=O)OC(C)(C)C",  # Boc2O
            "tert_butyl_carbamate": "[NX3]C(=O)OC(C)(C)C"  # t-Boc carbamate
        }
        
        self.deprotection_patterns = {
            "tfa_deprotection": "FC(F)(F)C(=O)O",  # TFA for Boc deprotection
            "hcl_deprotection": "[Cl-]",  # HCl conditions
            "acid_conditions": "[OH]C(=O)*"  # General acid patterns
        }

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        protection_events = []
        deprotection_events = []
        
        # Analyze each reaction for protection/deprotection events
        for i, rxn in enumerate(reactions):
            is_protection = self.detect_protection_reaction(rxn)
            is_deprotection = self.detect_deprotection_reaction(rxn)
            
            if is_protection:
                protection_events.append(i)
            if is_deprotection:
                deprotection_events.append(i)
        
        # Check if we have sequential strategy
        sequential_strategy = self.evaluate_sequential_strategy(
            protection_events, deprotection_events, len(reactions)
        )
        
        return sequential_strategy, len(reactions)
    
    def detect_protection_reaction(self, rxn) -> bool:
        """Detect if reaction involves Boc protection of amine"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Check for introduction of Boc group
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if not all(reactant_mols + product_mols):
                return False
            
            # Count Boc groups in reactants vs products
            reactant_boc_count = sum(
                len(mol.GetSubstructMatches(Chem.MolFromSmarts(self.boc_patterns["boc_protected_amine"])))
                for mol in reactant_mols if mol
            )
            
            product_boc_count = sum(
                len(mol.GetSubstructMatches(Chem.MolFromSmarts(self.boc_patterns["boc_protected_amine"])))
                for mol in product_mols if mol
            )
            
            # Protection: increase in Boc groups
            boc_increase = product_boc_count > reactant_boc_count
            
            # Check for Boc reagent presence
            boc_reagent_present = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.boc_patterns["boc_anhydride"]))
                for mol in reactant_mols if mol
            )
            
            return boc_increase or boc_reagent_present
            
        except Exception:
            return False
    
    def detect_deprotection_reaction(self, rxn) -> bool:
        """Detect if reaction involves Boc deprotection"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if not all(reactant_mols + product_mols):
                return False
            
            # Count Boc groups in reactants vs products
            reactant_boc_count = sum(
                len(mol.GetSubstructMatches(Chem.MolFromSmarts(self.boc_patterns["boc_protected_amine"])))
                for mol in reactant_mols if mol
            )
            
            product_boc_count = sum(
                len(mol.GetSubstructMatches(Chem.MolFromSmarts(self.boc_patterns["boc_protected_amine"])))
                for mol in product_mols if mol
            )
            
            # Deprotection: decrease in Boc groups
            boc_decrease = reactant_boc_count > product_boc_count
            
            # Check for deprotection reagents
            deprotection_reagent = any(
                any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for pattern in self.deprotection_patterns.values())
                for mol in reactant_mols if mol
            )
            
            return boc_decrease and deprotection_reagent
            
        except Exception:
            return False
    
    def evaluate_sequential_strategy(self, protection_events, deprotection_events, total_reactions) -> bool:
        """
        Evaluate if the protection/deprotection pattern represents a sequential strategy
        """
        if len(protection_events) == 0 or len(deprotection_events) == 0:
            return False
        
        # For sequential strategy, we expect:
        # 1. At least one protection followed by at least one deprotection
        # 2. Multiple protection/deprotection cycles OR
        # 3. Strategic timing (not all at once)
        
        # Check for proper sequencing
        has_protection_before_deprotection = any(
            prot < deprot for prot in protection_events for deprot in deprotection_events
        )
        
        if not has_protection_before_deprotection:
            return False
        
        # Multiple cycles indicate sequential strategy
        if len(protection_events) >= 2 or len(deprotection_events) >= 2:
            return True
        
        # Single protection/deprotection but with strategic spacing
        if len(protection_events) == 1 and len(deprotection_events) == 1:
            protection_step = protection_events[0]
            deprotection_step = deprotection_events[0]
            step_gap = deprotection_step - protection_step
            
            # Strategic if there are intervening reactions
            return step_gap > 1 and step_gap < total_reactions - 1
        
        return False
