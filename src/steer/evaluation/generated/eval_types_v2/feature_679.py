"""Generated evaluation code for: Sequential protecting group strategy for amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates sequential protecting group strategy for functional groups.
    Checks if a protecting group is added early in synthesis and removed late,
    with the functional group remaining protected during intermediate steps.
    """
    
    def __init__(self, config):
        self.functional_group = config["parameters"]["functional_group"]
        self.protecting_group = config["parameters"]["protecting_group"]
        self.strategy = config["parameters"]["strategy"]
        
        # Define SMARTS patterns for functional groups and protecting groups
        self.fg_patterns = {
            "amine": "[NX3;H2,H1;!$(NC=O)]",  # Primary or secondary amine, not amide
        }
        
        self.pg_patterns = {
            "Boc": "[NX3]C(=O)OC(C)(C)C",  # Boc-protected amine
        }
        
        self.fg_pattern = self.fg_patterns.get(self.functional_group)
        self.pg_pattern = self.pg_patterns.get(self.protecting_group)

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        total_steps = len(reactions)
        
        if total_steps < 2:
            return False, total_steps
            
        protection_step = -1
        deprotection_step = -1
        
        # Find protection and deprotection steps
        for i, rxn in enumerate(reactions):
            if self.is_protection_reaction(rxn):
                protection_step = i
            elif self.is_deprotection_reaction(rxn):
                deprotection_step = i
                
        # Check if strategy is followed
        condition_met = False
        
        if self.strategy == "early_protection_late_deprotection":
            if protection_step >= 0 and deprotection_step >= 0:
                # Protection should be early (first third), deprotection should be late (last third)
                early_threshold = total_steps // 3
                late_threshold = 2 * total_steps // 3
                
                protection_early = protection_step <= early_threshold
                deprotection_late = deprotection_step >= late_threshold
                protection_before_deprotection = protection_step < deprotection_step
                
                # Check that functional group remains protected between protection and deprotection
                protected_maintained = self.check_protection_maintained(reactions, protection_step, deprotection_step)
                
                condition_met = (protection_early and deprotection_late and 
                               protection_before_deprotection and protected_maintained)
        
        return condition_met, total_steps

    def is_protection_reaction(self, rxn):
        """Check if reaction introduces the protecting group"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            # Check if reactants have free functional group and products have protected group
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Count free functional groups in reactants
            free_fg_reactants = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(self.fg_pattern))) 
                                  for mol in reactant_mols if mol is not None)
            
            # Count protected groups in products
            protected_fg_products = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(self.pg_pattern))) 
                                      for mol in product_mols if mol is not None)
            
            # Protection reaction: free FG decreases, protected FG increases
            return free_fg_reactants > 0 and protected_fg_products > 0
            
        except:
            return False

    def is_deprotection_reaction(self, rxn):
        """Check if reaction removes the protecting group"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Count protected groups in reactants
            protected_fg_reactants = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(self.pg_pattern))) 
                                       for mol in reactant_mols if mol is not None)
            
            # Count free functional groups in products
            free_fg_products = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(self.fg_pattern))) 
                                 for mol in product_mols if mol is not None)
            
            # Deprotection reaction: protected FG decreases, free FG increases
            return protected_fg_reactants > 0 and free_fg_products > 0
            
        except:
            return False

    def check_protection_maintained(self, reactions, protection_step, deprotection_step):
        """Check that the protecting group is maintained between protection and deprotection"""
        for i in range(protection_step + 1, deprotection_step):
            rxn = reactions[i]
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                continue
                
            try:
                reactants = rxn_parts[0]
                products = rxn_parts[1]
                
                reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
                product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
                
                # Count protected groups before and after reaction
                protected_reactants = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(self.pg_pattern))) 
                                        for mol in reactant_mols if mol is not None)
                protected_products = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(self.pg_pattern))) 
                                       for mol in product_mols if mol is not None)
                
                # If protected group is lost in intermediate step, protection not maintained
                if protected_reactants > 0 and protected_products == 0:
                    return False
                    
            except:
                continue
                
        return True

    def route_scoring(self, condition_met_fraction):
        """Convert condition result to score"""
        if condition_met_fraction > 0:
            return 10.0  # Perfect score if strategy is followed
        else:
            return 0.0   # No score if strategy not followed
