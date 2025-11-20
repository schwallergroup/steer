"""Generated evaluation code for: Multiple ester functional group interconversions"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleEsterInterconversions(MultiRxnCondBase):
    """
    Evaluates routes based on multiple ester functional group interconversions.
    Checks for esterification, transesterification, and ester hydrolysis reactions
    occurring at the same position with minimum occurrence threshold.
    """
    
    def __init__(self, config):
        self.min_occurrences = config.get("min_occurrences", 4)
        self.involves_same_position = config.get("involves_same_position", True)
        
        # Define SMARTS patterns for ester-related reactions
        self.ester_patterns = {
            "ester": "[C:1](=[O:2])[O:3][C:4]",
            "carboxylic_acid": "[C:1](=[O:2])[OH:3]",
            "alcohol": "[C:1][OH:2]",
            "carbonyl": "[C:1]=[O:2]"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        ester_interconversions = []
        
        for rxn in reactions:
            interconversion_type = self.detect_ester_interconversion(rxn)
            if interconversion_type:
                ester_interconversions.append((interconversion_type, rxn))
        
        # Check if minimum occurrences met
        if len(ester_interconversions) < self.min_occurrences:
            return False, len(reactions)
        
        # If same position required, check atom map consistency
        if self.involves_same_position:
            same_position = self.check_same_position_transformations(ester_interconversions)
            if not same_position:
                return False, len(reactions)
        
        return True, len(reactions)
    
    def detect_ester_interconversion(self, rxn):
        """Detect if reaction is esterification, transesterification, or ester hydrolysis"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return None
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
        
        if not all(reactants + products):
            return None
        
        # Check for esterification (acid + alcohol -> ester + water)
        if self.is_esterification(reactants, products):
            return "esterification"
        
        # Check for ester hydrolysis (ester + water -> acid + alcohol)  
        if self.is_ester_hydrolysis(reactants, products):
            return "ester_hydrolysis"
        
        # Check for transesterification (ester1 + alcohol -> ester2 + alcohol)
        if self.is_transesterification(reactants, products):
            return "transesterification"
        
        return None
    
    def is_esterification(self, reactants, products):
        """Check if reaction is esterification"""
        acid_pattern = Chem.MolFromSmarts(self.ester_patterns["carboxylic_acid"])
        alcohol_pattern = Chem.MolFromSmarts(self.ester_patterns["alcohol"])
        ester_pattern = Chem.MolFromSmarts(self.ester_patterns["ester"])
        
        has_acid = any(mol.HasSubstructMatch(acid_pattern) for mol in reactants)
        has_alcohol = any(mol.HasSubstructMatch(alcohol_pattern) for mol in reactants)
        has_ester = any(mol.HasSubstructMatch(ester_pattern) for mol in products)
        
        return has_acid and has_alcohol and has_ester
    
    def is_ester_hydrolysis(self, reactants, products):
        """Check if reaction is ester hydrolysis"""
        acid_pattern = Chem.MolFromSmarts(self.ester_patterns["carboxylic_acid"])
        alcohol_pattern = Chem.MolFromSmarts(self.ester_patterns["alcohol"])
        ester_pattern = Chem.MolFromSmarts(self.ester_patterns["ester"])
        
        has_ester = any(mol.HasSubstructMatch(ester_pattern) for mol in reactants)
        has_acid = any(mol.HasSubstructMatch(acid_pattern) for mol in products)
        has_alcohol = any(mol.HasSubstructMatch(alcohol_pattern) for mol in products)
        
        return has_ester and has_acid and has_alcohol
    
    def is_transesterification(self, reactants, products):
        """Check if reaction is transesterification"""
        ester_pattern = Chem.MolFromSmarts(self.ester_patterns["ester"])
        alcohol_pattern = Chem.MolFromSmarts(self.ester_patterns["alcohol"])
        
        reactant_esters = sum(1 for mol in reactants if mol.HasSubstructMatch(ester_pattern))
        product_esters = sum(1 for mol in products if mol.HasSubstructMatch(ester_pattern))
        reactant_alcohols = sum(1 for mol in reactants if mol.HasSubstructMatch(alcohol_pattern))
        product_alcohols = sum(1 for mol in products if mol.HasSubstructMatch(alcohol_pattern))
        
        # Transesterification: ester + alcohol -> ester + alcohol
        return (reactant_esters >= 1 and product_esters >= 1 and 
                reactant_alcohols >= 1 and product_alcohols >= 1)
    
    def check_same_position_transformations(self, ester_interconversions):
        """Check if ester transformations occur at the same carbonyl position"""
        carbonyl_positions = []
        
        for interconversion_type, rxn in ester_interconversions:
            rxn_parts = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            
            for mol in reactants:
                if mol is None:
                    continue
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() > 0:
                        # Check if this mapped atom is part of carbonyl in ester/acid
                        if atom.GetSymbol() == 'C':
                            neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
                            if 'O' in neighbors and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'O']) >= 1:
                                carbonyl_positions.append(atom.GetAtomMapNum())
        
        # Check if at least 2 transformations involve the same carbonyl position
        if not carbonyl_positions:
            return False
            
        from collections import Counter
        position_counts = Counter(carbonyl_positions)
        return max(position_counts.values()) >= 2
