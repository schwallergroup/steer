"""Generated evaluation code for: Cyclic protecting group cycling pattern"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CyclicProtectingGroupCycling(MultiRxnCondBase):
    """
    Detects cyclic protecting group cycling patterns where protection is followed 
    by deprotection without meaningful intervening chemistry.
    
    Specifically looks for acetal formation followed by acetal deprotection
    on the same functional group with minimal intervening steps.
    """
    
    def __init__(self, config):
        self.protection_deprotection_pairs = config.get("protection_deprotection_pairs", ["acetal_formation", "acetal_deprotection"])
        self.max_intervening_steps = config.get("intervening_steps", 0)
        self.same_functional_group = config.get("same_functional_group", True)
        
        # SMARTS patterns for acetal groups
        self.acetal_pattern = Chem.MolFromSmarts("[CH1]([OR])([OR])")  # Acetal carbon with two OR groups
        self.ketone_pattern = Chem.MolFromSmarts("[CX3](=[OX1])")  # Ketone carbonyl
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        total_reactions = len(reactions)
        
        # Look for protection-deprotection cycles
        has_cycling = self.detect_protection_cycling(reactions)
        
        return has_cycling, total_reactions
    
    def detect_protection_cycling(self, reactions) -> bool:
        """Detect if there's a cyclic protection-deprotection pattern"""
        protection_steps = []
        deprotection_steps = []
        
        # Identify protection and deprotection steps
        for i, rxn in enumerate(reactions):
            if self.is_acetal_formation(rxn):
                protection_steps.append(i)
            elif self.is_acetal_deprotection(rxn):
                deprotection_steps.append(i)
        
        # Check for cycling patterns
        for prot_idx in protection_steps:
            for deprot_idx in deprotection_steps:
                if deprot_idx > prot_idx:  # Deprotection after protection
                    intervening = deprot_idx - prot_idx - 1
                    
                    if intervening <= self.max_intervening_steps:
                        # Check if same functional group is involved
                        if self.same_functional_group:
                            if self.same_group_protected_deprotected(reactions[prot_idx], reactions[deprot_idx]):
                                return True
                        else:
                            return True
        
        return False
    
    def is_acetal_formation(self, rxn) -> bool:
        """Check if reaction involves acetal formation (ketone -> acetal)"""
        try:
            rxn_parts = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
            
            # Count ketones in reactants and acetals in products
            reactant_ketones = sum(len(mol.GetSubstructMatches(self.ketone_pattern)) 
                                 for mol in reactants if mol is not None)
            product_acetals = sum(len(mol.GetSubstructMatches(self.acetal_pattern)) 
                                for mol in products if mol is not None)
            
            # Acetal formation: ketones decrease, acetals increase
            reactant_acetals = sum(len(mol.GetSubstructMatches(self.acetal_pattern)) 
                                 for mol in reactants if mol is not None)
            product_ketones = sum(len(mol.GetSubstructMatches(self.ketone_pattern)) 
                                for mol in products if mol is not None)
            
            return (reactant_ketones > product_ketones) and (product_acetals > reactant_acetals)
            
        except:
            return False
    
    def is_acetal_deprotection(self, rxn) -> bool:
        """Check if reaction involves acetal deprotection (acetal -> ketone)"""
        try:
            rxn_parts = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
            
            # Count acetals in reactants and ketones in products
            reactant_acetals = sum(len(mol.GetSubstructMatches(self.acetal_pattern)) 
                                 for mol in reactants if mol is not None)
            product_ketones = sum(len(mol.GetSubstructMatches(self.ketone_pattern)) 
                                for mol in products if mol is not None)
            
            # Acetal deprotection: acetals decrease, ketones increase
            reactant_ketones = sum(len(mol.GetSubstructMatches(self.ketone_pattern)) 
                                 for mol in reactants if mol is not None)
            product_acetals = sum(len(mol.GetSubstructMatches(self.acetal_pattern)) 
                                for mol in products if mol is not None)
            
            return (reactant_acetals > product_acetals) and (product_ketones > reactant_ketones)
            
        except:
            return False
    
    def same_group_protected_deprotected(self, protection_rxn, deprotection_rxn) -> bool:
        """Check if the same functional group is protected and then deprotected"""
        try:
            # Extract atom mapping numbers for protected carbons
            prot_parts = protection_rxn.split(">>")
            deprot_parts = deprotection_rxn.split(">>")
            
            # Find ketone carbons that become acetals in protection
            prot_reactants = [Chem.MolFromSmiles(r) for r in prot_parts[0].split(".")]
            prot_products = [Chem.MolFromSmiles(p) for p in prot_parts[1].split(".")]
            
            protected_atoms = set()
            for mol in prot_reactants:
                if mol is not None:
                    for match in mol.GetSubstructMatches(self.ketone_pattern):
                        ketone_carbon = mol.GetAtomWithIdx(match[0])
                        if ketone_carbon.GetAtomMapNum() > 0:
                            protected_atoms.add(ketone_carbon.GetAtomMapNum())
            
            # Find acetal carbons that become ketones in deprotection
            deprot_reactants = [Chem.MolFromSmiles(r) for r in deprot_parts[0].split(".")]
            deprotected_atoms = set()
            for mol in deprot_reactants:
                if mol is not None:
                    for match in mol.GetSubstructMatches(self.acetal_pattern):
                        acetal_carbon = mol.GetAtomWithIdx(match[0])
                        if acetal_carbon.GetAtomMapNum() > 0:
                            deprotected_atoms.add(acetal_carbon.GetAtomMapNum())
            
            return len(protected_atoms.intersection(deprotected_atoms)) > 0
            
        except:
            return False
