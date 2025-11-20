"""Generated evaluation code for: Acetate protecting group cycling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AcetateProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates synthesis routes for acetate protecting group cycling strategy.
    Checks if a secondary alcohol is protected with acetate at the specified step
    and deprotected at another specified step.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "acetate")
        self.protection_step = config.get("protection_step", 5)
        self.deprotection_step = config.get("deprotection_step", 2)
        self.group_protected = config.get("group_protected", "secondary_alcohol")
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        total_steps = len(reactions)
        
        # Check if we have enough steps for both protection and deprotection
        if total_steps < max(self.protection_step, self.deprotection_step):
            return False, total_steps
        
        protection_found = False
        deprotection_found = False
        
        # Check for protection step (acetylation of secondary alcohol)
        if self.protection_step <= total_steps:
            protection_rxn = reactions[self.protection_step - 1]
            protection_found = self.detect_acetate_protection(protection_rxn)
        
        # Check for deprotection step (acetate removal)
        if self.deprotection_step <= total_steps:
            deprotection_rxn = reactions[self.deprotection_step - 1]
            deprotection_found = self.detect_acetate_deprotection(deprotection_rxn)
        
        condition = protection_found and deprotection_found
        return condition, total_steps
    
    def detect_acetate_protection(self, rxn):
        """Detect acetylation of secondary alcohol"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Pattern for secondary alcohol
            sec_alcohol_pattern = Chem.MolFromSmarts("[CH1]([#6])[OH1]")
            # Pattern for acetate ester
            acetate_pattern = Chem.MolFromSmarts("[CH1]([#6])OC(=O)C")
            
            # Check if reactants contain secondary alcohol
            has_sec_alcohol_reactant = any(mol.HasSubstructMatch(sec_alcohol_pattern) 
                                         for mol in reactants)
            
            # Check if products contain acetate ester
            has_acetate_product = any(mol.HasSubstructMatch(acetate_pattern) 
                                    for mol in products)
            
            # Check for acetylating agent (acetyl chloride, acetic anhydride, etc.)
            acetylating_agents = [
                Chem.MolFromSmarts("CC(=O)Cl"),  # acetyl chloride
                Chem.MolFromSmarts("CC(=O)OC(=O)C"),  # acetic anhydride
            ]
            
            has_acetylating_agent = any(
                any(mol.HasSubstructMatch(agent) for mol in reactants)
                for agent in acetylating_agents if agent
            )
            
            return has_sec_alcohol_reactant and has_acetate_product and has_acetylating_agent
            
        except Exception:
            return False
    
    def detect_acetate_deprotection(self, rxn):
        """Detect removal of acetate protecting group"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Pattern for acetate ester
            acetate_pattern = Chem.MolFromSmarts("[CH1]([#6])OC(=O)C")
            # Pattern for secondary alcohol
            sec_alcohol_pattern = Chem.MolFromSmarts("[CH1]([#6])[OH1]")
            
            # Check if reactants contain acetate ester
            has_acetate_reactant = any(mol.HasSubstructMatch(acetate_pattern) 
                                     for mol in reactants)
            
            # Check if products contain secondary alcohol
            has_sec_alcohol_product = any(mol.HasSubstructMatch(sec_alcohol_pattern) 
                                        for mol in products)
            
            # Check for deprotection conditions (base or acid)
            deprotection_reagents = [
                Chem.MolFromSmarts("[OH-]"),  # hydroxide
                Chem.MolFromSmarts("[K+].[OH-]"),  # KOH
                Chem.MolFromSmarts("[Na+].[OH-]"),  # NaOH
                Chem.MolFromSmarts("N(C)(C)C"),  # trimethylamine
            ]
            
            has_deprotection_reagent = any(
                any(mol.HasSubstructMatch(reagent) for mol in reactants)
                for reagent in deprotection_reagents if reagent
            )
            
            return has_acetate_reactant and has_sec_alcohol_product
            
        except Exception:
            return False
