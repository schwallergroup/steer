"""Generated evaluation code for: Orthogonal protecting group strategy with four groups"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class OrthogonalProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates whether a synthesis route uses an orthogonal protecting group strategy
    with specified protecting groups that can be removed selectively.
    """
    
    def __init__(self, config):
        self.required_groups = config["groups"]
        self.orthogonal = config.get("orthogonal", True)
        
        # Define SMARTS patterns for each protecting group
        self.protecting_group_patterns = {
            "Boc": "[NX3][CX3](=[OX1])[OX2][CX4]([CH3])([CH3])[CH3]",  # tert-butoxycarbonyl
            "allyl": "[CH2]=[CH1][CH2][OX2,NX3]",  # allyl ether/amine
            "benzyl": "c1ccccc1[CH2][OX2,NX3]",  # benzyl ether/amine
            "TBS": "[Si]([CH3])([CH3])[CX4]([CH3])([CH3])[CH3]",  # tert-butyldimethylsilyl
            "Cbz": "[NX3][CX3](=[OX1])[OX2][CH2]c1ccccc1",  # carbobenzyloxy
            "Fmoc": "[NX3][CX3](=[OX1])[OX2][CH2][CH1]1c2ccccc2c3ccccc13",  # fluorenylmethoxycarbonyl
            "Ac": "[NX3,OX2][CX3](=[OX1])[CH3]",  # acetyl
            "TIPS": "[Si]([CH1]([CH3])[CH3])([CH1]([CH3])[CH3])[CH1]([CH3])[CH3]"  # triisopropylsilyl
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Check if the route uses the required orthogonal protecting group strategy.
        Returns (condition_met, total_reactions).
        """
        reactions = self.get_rxns(d)
        
        # Track protecting group installations and removals
        installed_groups = set()
        removed_groups = set()
        
        for rxn in reactions:
            # Check for protecting group installation
            for group_name in self.required_groups:
                if self.detect_protection_installation(rxn, group_name):
                    installed_groups.add(group_name)
                
                # Check for protecting group removal
                if self.detect_protection_removal(rxn, group_name):
                    removed_groups.add(group_name)
        
        # Check if all required groups were used
        all_groups_used = all(group in installed_groups for group in self.required_groups)
        
        # Check orthogonality - at least some groups should be selectively removed
        orthogonal_removal = len(removed_groups) >= 2 if self.orthogonal else True
        
        # Ensure we have at least the minimum number of required groups
        sufficient_diversity = len(installed_groups) >= len(self.required_groups)
        
        condition = all_groups_used and orthogonal_removal and sufficient_diversity
        
        return condition, len(reactions)
    
    def detect_protection_installation(self, rxn, group_name):
        """
        Detect if a protecting group is installed in this reaction.
        """
        if group_name not in self.protecting_group_patterns:
            return False
            
        pattern = self.protecting_group_patterns[group_name]
        
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            # Remove None molecules (parsing failures)
            reactants = [m for m in reactants if m is not None]
            products = [m for m in products if m is not None]
            
            # Count protecting group occurrences
            reactant_matches = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))) 
                                 for mol in reactants)
            product_matches = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))) 
                                for mol in products)
            
            # Protection installation: more protecting groups in products than reactants
            return product_matches > reactant_matches
            
        except:
            return False
    
    def detect_protection_removal(self, rxn, group_name):
        """
        Detect if a protecting group is removed in this reaction.
        """
        if group_name not in self.protecting_group_patterns:
            return False
            
        pattern = self.protecting_group_patterns[group_name]
        
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            # Remove None molecules (parsing failures)
            reactants = [m for m in reactants if m is not None]
            products = [m for m in products if m is not None]
            
            # Count protecting group occurrences
            reactant_matches = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))) 
                                 for mol in reactants)
            product_matches = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))) 
                                for mol in products)
            
            # Protection removal: fewer protecting groups in products than reactants
            return reactant_matches > product_matches
            
        except:
            return False
