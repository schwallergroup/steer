"""Generated evaluation code for: Benzyl protecting group with subsequent deprotection"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes for proper benzyl protecting group strategy.
    Checks for installation of N-benzyl protection followed by deprotection
    at specified steps in the synthesis.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "benzyl")
        self.protection_step = config.get("protection_step", 4)
        self.deprotection_step = config.get("deprotection_step", 3)
        self.functional_group = config.get("functional_group", "amine")
        
        # Define SMARTS patterns for benzyl protection/deprotection
        self.benzyl_amine_pattern = Chem.MolFromSmarts("[CH2]c1ccccc1-[N]")
        self.free_amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        protection_found = False
        deprotection_found = False
        protection_at_correct_step = False
        deprotection_at_correct_step = False
        
        for i, rxn in enumerate(reactions):
            step_number = i + 1
            
            # Check for benzyl protection (amine + benzyl reagent -> benzyl-protected amine)
            if self.detect_benzyl_protection(rxn):
                protection_found = True
                if step_number == self.protection_step:
                    protection_at_correct_step = True
                    
            # Check for debenzylation (benzyl-protected amine -> free amine)
            if self.detect_benzyl_deprotection(rxn):
                deprotection_found = True
                if step_number == self.deprotection_step:
                    deprotection_at_correct_step = True
        
        # Condition is met if both protection and deprotection occur at correct steps
        condition = (protection_found and deprotection_found and 
                    protection_at_correct_step and deprotection_at_correct_step)
        
        return condition, len(reactions)
    
    def detect_benzyl_protection(self, rxn):
        """Detect formation of N-benzyl bond from free amine"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
        
        if not all(reactants) or not all(products):
            return False
        
        # Check if reactants contain free amine and products contain benzyl-protected amine
        has_free_amine_reactant = any(mol.HasSubstructMatch(self.free_amine_pattern) for mol in reactants)
        has_benzyl_amine_product = any(mol.HasSubstructMatch(self.benzyl_amine_pattern) for mol in products)
        
        # Also check for benzyl halide or benzyl alcohol in reactants
        benzyl_reagent_pattern = Chem.MolFromSmarts("[CH2]c1ccccc1[Cl,Br,I,OH]")
        has_benzyl_reagent = any(mol.HasSubstructMatch(benzyl_reagent_pattern) for mol in reactants)
        
        return has_free_amine_reactant and has_benzyl_amine_product and has_benzyl_reagent
    
    def detect_benzyl_deprotection(self, rxn):
        """Detect cleavage of N-benzyl bond to give free amine"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
        
        if not all(reactants) or not all(products):
            return False
        
        # Check if reactants contain benzyl-protected amine and products contain free amine
        has_benzyl_amine_reactant = any(mol.HasSubstructMatch(self.benzyl_amine_pattern) for mol in reactants)
        has_free_amine_product = any(mol.HasSubstructMatch(self.free_amine_pattern) for mol in products)
        
        # Common debenzylation conditions: H2/Pd, BCl3, etc.
        debenzylation_reagent_patterns = [
            Chem.MolFromSmarts("[H][H]"),  # H2
            Chem.MolFromSmarts("B(Cl)(Cl)Cl"),  # BCl3
            Chem.MolFromSmarts("[Pd]")  # Pd catalyst
        ]
        
        has_debenzylation_reagent = any(
            any(mol.HasSubstructMatch(pattern) for mol in reactants)
            for pattern in debenzylation_reagent_patterns
        )
        
        return has_benzyl_amine_reactant and has_free_amine_product
