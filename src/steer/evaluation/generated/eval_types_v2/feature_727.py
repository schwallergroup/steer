"""Generated evaluation code for: Multiple ester hydrolysis-reformation cycles"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleEsterCycles(MultiRxnCondBase):
    """
    Detects multiple ester hydrolysis-reformation cycles in a synthesis route.
    Identifies routes with repeated ester protection/deprotection cycles without strategic purpose.
    """
    
    def __init__(self, config):
        self.cycle_count = config.get("cycle_count", 2)
        self.sequence = config.get("sequence", ["methyl_ester_to_acid", "acid_to_ethyl_ester", "ethyl_ester_to_acid"])
        
        # SMARTS patterns for ester hydrolysis and esterification
        self.methyl_ester_pattern = Chem.MolFromSmarts("C(=O)OC")
        self.ethyl_ester_pattern = Chem.MolFromSmarts("C(=O)OCC")
        self.carboxylic_acid_pattern = Chem.MolFromSmarts("C(=O)O")
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track ester transformations throughout the route
        ester_transformations = []
        
        for rxn in reactions:
            transformation_type = self.classify_ester_transformation(rxn)
            if transformation_type:
                ester_transformations.append(transformation_type)
        
        # Check if we have the specified sequence pattern repeated
        cycles_found = self.count_transformation_cycles(ester_transformations)
        
        condition = cycles_found >= self.cycle_count
        return condition, len(reactions)
    
    def classify_ester_transformation(self, rxn):
        """Classify the type of ester transformation occurring in the reaction."""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return None
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
            
            if not all(reactants) or not all(products):
                return None
            
            # Check for methyl ester to acid transformation
            if (any(mol.HasSubstructMatch(self.methyl_ester_pattern) for mol in reactants) and
                any(mol.HasSubstructMatch(self.carboxylic_acid_pattern) for mol in products) and
                not any(mol.HasSubstructMatch(self.methyl_ester_pattern) for mol in products)):
                return "methyl_ester_to_acid"
            
            # Check for acid to ethyl ester transformation
            if (any(mol.HasSubstructMatch(self.carboxylic_acid_pattern) for mol in reactants) and
                any(mol.HasSubstructMatch(self.ethyl_ester_pattern) for mol in products) and
                not any(mol.HasSubstructMatch(self.carboxylic_acid_pattern) for mol in products)):
                return "acid_to_ethyl_ester"
            
            # Check for ethyl ester to acid transformation
            if (any(mol.HasSubstructMatch(self.ethyl_ester_pattern) for mol in reactants) and
                any(mol.HasSubstructMatch(self.carboxylic_acid_pattern) for mol in products) and
                not any(mol.HasSubstructMatch(self.ethyl_ester_pattern) for mol in products)):
                return "ethyl_ester_to_acid"
            
            return None
            
        except Exception:
            return None
    
    def count_transformation_cycles(self, transformations):
        """Count how many complete cycles of the specified sequence occur."""
        if len(transformations) < len(self.sequence):
            return 0
        
        cycles = 0
        i = 0
        
        while i <= len(transformations) - len(self.sequence):
            # Check if sequence starting at position i matches our target sequence
            matches_sequence = True
            for j, expected_transformation in enumerate(self.sequence):
                if i + j >= len(transformations) or transformations[i + j] != expected_transformation:
                    matches_sequence = False
                    break
            
            if matches_sequence:
                cycles += 1
                i += len(self.sequence)  # Move past this complete cycle
            else:
                i += 1
        
        return cycles
