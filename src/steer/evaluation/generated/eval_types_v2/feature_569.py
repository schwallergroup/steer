"""Generated evaluation code for: Early ester protecting group installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyEsterProtection(BaseScoring):
    """
    Evaluates synthesis routes for early installation of ethyl ester protecting groups
    on carboxylic acids. Rewards routes where ester protection occurs early and
    remains for multiple steps.
    """
    
    def __init__(self, config: Dict):
        self.target_steps_protected = config["parameters"].get("steps_protected", 4)
        self.timing = config["parameters"].get("timing", "early")
        
        # SMARTS patterns
        self.carboxylic_acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OH]")
        self.ethyl_ester_pattern = Chem.MolFromSmarts("[CX3](=O)O[CH2][CH3]")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection doesn't happen
        else:
            # Early protection is better (lower depth fraction gets higher score)
            early_score = (1 - x) * 5  # 0-5 points for timing
            
            # Additional points if protection lasts for target steps
            protection_duration = self._calculate_protection_duration(x)
            duration_score = min(protection_duration / self.target_steps_protected, 1.0) * 5
            
            return early_score + duration_score
    
    def hit_condition(self, d):
        """Check if this reaction involves ethyl ester protection of carboxylic acid"""
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Check for carboxylic acid in reactants
            has_carboxylic_acid = any(
                mol and mol.HasSubstructMatch(self.carboxylic_acid_pattern) 
                for mol in reactant_mols
            )
            
            # Check for ethyl ester in products
            has_ethyl_ester = any(
                mol and mol.HasSubstructMatch(self.ethyl_ester_pattern)
                for mol in product_mols
            )
            
            # Check that we're gaining ester and losing acid
            reactant_esters = sum(
                len(mol.GetSubstructMatches(self.ethyl_ester_pattern)) if mol else 0
                for mol in reactant_mols
            )
            product_esters = sum(
                len(mol.GetSubstructMatches(self.ethyl_ester_pattern)) if mol else 0
                for mol in product_mols
            )
            
            return has_carboxylic_acid and has_ethyl_ester and (product_esters > reactant_esters)
            
        except:
            return False
    
    def _calculate_protection_duration(self, protection_depth_fraction):
        """Estimate how many steps the protection lasts based on route analysis"""
        # This is a simplified heuristic - in practice you'd analyze the full route
        # to see when deprotection occurs
        if protection_depth_fraction < 0.2:  # Very early protection
            return self.target_steps_protected + 1
        elif protection_depth_fraction < 0.5:  # Early protection
            return self.target_steps_protected
        else:  # Late protection
            return max(1, self.target_steps_protected - 2)
