"""Generated evaluation code for: Boc protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectingGroupStrategy(BaseScoring):
    """
    Evaluates synthesis routes based on Boc protecting group strategy.
    Checks if Boc protection occurs early and deprotection occurs at the specified step.
    """
    
    def __init__(self, config: Dict):
        self.target_deprotection_step = config["parameters"]["deprotection_step"]
        self.protecting_group = config["parameters"]["protecting_group"]
        self.functional_group = config["parameters"]["functional_group"]
        
        # SMARTS patterns for Boc group and amine
        self.boc_pattern = Chem.MolFromSmarts("[CH3][C](=O)[O][C]([CH3])([CH3])[CH3]")  # Boc group
        self.protected_amine_pattern = Chem.MolFromSmarts("[NH1][C](=O)[O][C]([CH3])([CH3])[CH3]")  # Boc-protected amine
        self.free_amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")  # Free amine
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No proper Boc strategy found
        else:
            # Score based on how close deprotection is to target step
            step_difference = abs(x - self.target_deprotection_step)
            if step_difference == 0:
                return 10  # Perfect timing
            elif step_difference <= 2:
                return 8 - step_difference  # Good timing
            else:
                return max(0, 6 - step_difference)  # Suboptimal timing
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Check if proper Boc protecting group strategy is used:
        1. Boc protection occurs early in the synthesis
        2. Deprotection occurs at the specified step
        """
        reactions = []
        self._collect_reactions(d, reactions, 0)
        
        protection_step = -1
        deprotection_step = -1
        
        for step, reaction in enumerate(reactions):
            if self._is_boc_protection(reaction):
                if protection_step == -1:  # Record first protection
                    protection_step = step
            elif self._is_boc_deprotection(reaction):
                deprotection_step = step
        
        # Strategy is valid if both protection and deprotection are found
        # and protection occurs before deprotection
        if protection_step >= 0 and deprotection_step >= 0 and protection_step < deprotection_step:
            return True, deprotection_step
        
        return False, -1
    
    def _collect_reactions(self, node, reactions, depth):
        """Recursively collect all reactions in the synthesis tree."""
        if "metadata" in node and "mapped_reaction_smiles" in node["metadata"]:
            reactions.append((depth, node["metadata"]["mapped_reaction_smiles"]))
        
        for child in node.get("children", []):
            self._collect_reactions(child, reactions, depth + 1)
    
    def _is_boc_protection(self, reaction_data):
        """Check if reaction involves Boc protection of an amine."""
        if isinstance(reaction_data, tuple):
            _, reaction_smiles = reaction_data
        else:
            reaction_smiles = reaction_data
            
        rxn_parts = reaction_smiles.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        # Check if reactants contain free amine and Boc reagent
        has_free_amine = any(mol and mol.HasSubstructMatch(self.free_amine_pattern) for mol in reactants)
        has_boc_reagent = any(mol and mol.HasSubstructMatch(self.boc_pattern) for mol in reactants)
        
        # Check if products contain Boc-protected amine
        has_protected_amine = any(mol and mol.HasSubstructMatch(self.protected_amine_pattern) for mol in products)
        
        return has_free_amine and (has_boc_reagent or has_protected_amine)
    
    def _is_boc_deprotection(self, reaction_data):
        """Check if reaction involves Boc deprotection."""
        if isinstance(reaction_data, tuple):
            _, reaction_smiles = reaction_data
        else:
            reaction_smiles = reaction_data
            
        rxn_parts = reaction_smiles.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        # Check if reactants contain Boc-protected amine
        has_protected_amine = any(mol and mol.HasSubstructMatch(self.protected_amine_pattern) for mol in reactants)
        
        # Check if products contain free amine and fewer Boc groups
        has_free_amine = any(mol and mol.HasSubstructMatch(self.free_amine_pattern) for mol in products)
        
        return has_protected_amine and has_free_amine
