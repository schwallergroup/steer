"""Generated evaluation code for: Sequential ester group conversion"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialEsterConversion(MultiRxnCondBase):
    """
    Evaluates whether a synthesis route contains sequential ester group conversion:
    deprotection followed by esterification in consecutive steps.
    """
    
    def __init__(self, config):
        self.reaction_sequence = config.get("reaction_sequence", ["deprotection", "esterification"])
        self.functional_group = config.get("functional_group", "ester")
        self.consecutive = config.get("consecutive", True)
        
        # Define SMARTS patterns for ester groups
        self.ester_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][CX4]")
        self.benzyl_ester_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2]Cc1ccccc1")
        self.carboxylic_acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2H]")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if sequential ester conversion occurs in the route."""
        reactions = self.get_rxns(d)
        
        if len(reactions) < 2:
            return False, len(reactions)
        
        # Find deprotection and esterification reactions
        deprotection_indices = []
        esterification_indices = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_deprotection(rxn):
                deprotection_indices.append(i)
            if self.detect_esterification(rxn):
                esterification_indices.append(i)
        
        # Check for consecutive sequence
        if self.consecutive:
            for dep_idx in deprotection_indices:
                if (dep_idx + 1) in esterification_indices:
                    return True, len(reactions)
        else:
            # Check if both reaction types are present (not necessarily consecutive)
            if deprotection_indices and esterification_indices:
                return True, len(reactions)
        
        return False, len(reactions)
    
    def detect_deprotection(self, rxn):
        """Detect ester deprotection reactions (e.g., benzyl ester to carboxylic acid)."""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check if reactant has protected ester (e.g., benzyl ester)
            has_protected_ester = any(mol.HasSubstructMatch(self.benzyl_ester_pattern) for mol in reactants)
            
            # Check if product has free carboxylic acid
            has_free_acid = any(mol.HasSubstructMatch(self.carboxylic_acid_pattern) for mol in products)
            
            return has_protected_ester and has_free_acid
            
        except:
            return False
    
    def detect_esterification(self, rxn):
        """Detect esterification reactions (carboxylic acid to ester)."""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check if reactant has carboxylic acid
            has_acid = any(mol.HasSubstructMatch(self.carboxylic_acid_pattern) for mol in reactants)
            
            # Check if product has ester
            has_ester = any(mol.HasSubstructMatch(self.ester_pattern) for mol in products)
            
            # Exclude cases where it's just deprotection (should form new ester bond)
            reactant_ester_count = sum(len(mol.GetSubstructMatches(self.ester_pattern)) for mol in reactants)
            product_ester_count = sum(len(mol.GetSubstructMatches(self.ester_pattern)) for mol in products)
            
            return has_acid and has_ester and product_ester_count > reactant_ester_count
            
        except:
            return False
