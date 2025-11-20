"""Generated evaluation code for: Circular ester-nitrile functional group interconversion"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CircularEsterNitrileInterconversion(MultiRxnCondBase):
    """
    Detects circular ester-nitrile functional group interconversion patterns.
    Checks if the route contains a circular pattern of esterification,
    deesterification, nitrile formation, and nitrile hydrolysis reactions.
    """
    
    def __init__(self, config):
        self.required_reactions = config.get("reaction_types", [
            "esterification", "deesterification", "nitrile_formation", "nitrile_hydrolysis"
        ])
        self.min_cycle_length = config.get("min_cycle_length", 3)
        
        # Define SMARTS patterns for functional groups
        self.ester_pattern = Chem.MolFromSmarts("[#6](=[#8])-[#8]-[#6]")  # C(=O)-O-C
        self.nitrile_pattern = Chem.MolFromSmarts("[#6]#[#7]")  # C#N
        self.amide_pattern = Chem.MolFromSmarts("[#6](=[#8])-[#7]")  # C(=O)-N
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Classify each reaction
        reaction_sequence = []
        for rxn in reactions:
            rxn_type = self.classify_reaction(rxn)
            if rxn_type:
                reaction_sequence.append(rxn_type)
        
        # Check for circular pattern
        has_circular_pattern = self.detect_circular_pattern(reaction_sequence)
        
        return has_circular_pattern, len(reactions)
    
    def classify_reaction(self, rxn):
        """Classify reaction based on functional group changes"""
        try:
            rxn_parts = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            products = [Chem.MolFromSmiles(p) for p in rxn_parts[0].split(".")]
            
            # Count functional groups in reactants and products
            reactant_esters = sum(self.count_functional_group(mol, self.ester_pattern) for mol in reactants if mol)
            product_esters = sum(self.count_functional_group(mol, self.ester_pattern) for mol in products if mol)
            
            reactant_nitriles = sum(self.count_functional_group(mol, self.nitrile_pattern) for mol in reactants if mol)
            product_nitriles = sum(self.count_functional_group(mol, self.nitrile_pattern) for mol in products if mol)
            
            reactant_amides = sum(self.count_functional_group(mol, self.amide_pattern) for mol in reactants if mol)
            product_amides = sum(self.count_functional_group(mol, self.amide_pattern) for mol in products if mol)
            
            # Classify based on functional group changes
            if product_esters > reactant_esters:
                return "esterification"
            elif reactant_esters > product_esters:
                return "deesterification"
            elif product_nitriles > reactant_nitriles:
                return "nitrile_formation"
            elif reactant_nitriles > product_nitriles:
                return "nitrile_hydrolysis"
            
            return None
            
        except Exception:
            return None
    
    def count_functional_group(self, mol, pattern):
        """Count occurrences of a functional group pattern in a molecule"""
        if mol is None or pattern is None:
            return 0
        return len(mol.GetSubstructMatches(pattern))
    
    def detect_circular_pattern(self, reaction_sequence):
        """Detect if the reaction sequence contains a circular pattern"""
        if len(reaction_sequence) < self.min_cycle_length:
            return False
        
        # Look for cycles where we return to the same functional group
        # A circular pattern would involve: ester -> amide -> nitrile -> ester (or variations)
        target_reactions = set(self.required_reactions)
        found_reactions = set(reaction_sequence)
        
        # Check if we have all required reaction types
        if not target_reactions.issubset(found_reactions):
            return False
        
        # Check for alternating pattern indicating circular interconversion
        # Look for sequences where esterification and deesterification both occur
        # along with nitrile formation and hydrolysis
        has_ester_cycle = "esterification" in reaction_sequence and "deesterification" in reaction_sequence
        has_nitrile_cycle = "nitrile_formation" in reaction_sequence and "nitrile_hydrolysis" in reaction_sequence
        
        return has_ester_cycle and has_nitrile_cycle
