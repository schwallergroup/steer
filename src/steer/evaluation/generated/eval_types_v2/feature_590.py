"""Generated evaluation code for: Sequential ester protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialEsterProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates synthesis routes for sequential ester protecting group cycling strategy.
    Checks for the presence of methyl ester -> acid -> t-butyl ester -> acid -> methyl ester
    transformations in the specified sequence.
    """
    
    def __init__(self, config):
        self.protecting_groups = config["parameters"]["protecting_groups"]
        self.sequence_type = config["parameters"]["sequence_type"]
        self.functional_group = config["parameters"]["functional_group"]
        
        # Define SMARTS patterns for different ester types and carboxylic acids
        self.methyl_ester_pattern = "[#6][C](=[O])[O][CH3]"
        self.tert_butyl_ester_pattern = "[#6][C](=[O])[O]C(C)(C)C"
        self.carboxylic_acid_pattern = "[#6][C](=[O])[OH]"
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track the sequence of protecting group transformations
        transformation_sequence = []
        
        for rxn in reactions:
            transformation = self.identify_transformation(rxn)
            if transformation:
                transformation_sequence.append(transformation)
        
        # Check if the sequence matches the expected cycling pattern
        condition = self.matches_cycling_pattern(transformation_sequence)
        
        return condition, len(reactions)
    
    def identify_transformation(self, rxn):
        """Identify the type of protecting group transformation in a reaction"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return None
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Parse reactants and products
        reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".") if smi.strip()]
        product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".") if smi.strip()]
        
        if not reactant_mols or not product_mols:
            return None
        
        # Check for different transformation types
        reactant_has_methyl = any(self.has_substructure(mol, self.methyl_ester_pattern) for mol in reactant_mols if mol)
        reactant_has_tbutyl = any(self.has_substructure(mol, self.tert_butyl_ester_pattern) for mol in reactant_mols if mol)
        reactant_has_acid = any(self.has_substructure(mol, self.carboxylic_acid_pattern) for mol in reactant_mols if mol)
        
        product_has_methyl = any(self.has_substructure(mol, self.methyl_ester_pattern) for mol in product_mols if mol)
        product_has_tbutyl = any(self.has_substructure(mol, self.tert_butyl_ester_pattern) for mol in product_mols if mol)
        product_has_acid = any(self.has_substructure(mol, self.carboxylic_acid_pattern) for mol in product_mols if mol)
        
        # Identify transformation type
        if reactant_has_methyl and product_has_acid:
            return "methyl_to_acid"
        elif reactant_has_acid and product_has_methyl:
            return "acid_to_methyl"
        elif reactant_has_tbutyl and product_has_acid:
            return "tbutyl_to_acid"
        elif reactant_has_acid and product_has_tbutyl:
            return "acid_to_tbutyl"
        elif reactant_has_methyl and product_has_tbutyl:
            return "methyl_to_tbutyl"
        elif reactant_has_tbutyl and product_has_methyl:
            return "tbutyl_to_methyl"
        
        return None
    
    def has_substructure(self, mol, pattern):
        """Check if molecule contains the specified substructure pattern"""
        if mol is None:
            return False
        try:
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol is None:
                return False
            return mol.HasSubstructMatch(pattern_mol)
        except:
            return False
    
    def matches_cycling_pattern(self, transformations):
        """Check if transformations match the expected cycling pattern"""
        if not transformations:
            return False
        
        # Expected pattern: methyl ester -> acid -> t-butyl ester -> acid -> methyl ester
        expected_cycle = [
            "methyl_to_acid",
            "acid_to_tbutyl", 
            "tbutyl_to_acid",
            "acid_to_methyl"
        ]
        
        # Check if we can find the complete cycle or significant portions
        cycle_matches = 0
        cycle_position = 0
        
        for transformation in transformations:
            if cycle_position < len(expected_cycle) and transformation == expected_cycle[cycle_position]:
                cycle_position += 1
                if cycle_position == len(expected_cycle):
                    cycle_matches += 1
                    cycle_position = 0  # Reset for potential additional cycles
        
        # Also check for partial cycles (at least 3 consecutive steps)
        partial_cycle_found = cycle_position >= 3
        
        return cycle_matches > 0 or partial_cycle_found
    
    def route_scoring(self, x):
        """Convert condition result to score"""
        if x < 0:
            return 0  # Pattern not found
        else:
            return 10  # Pattern found, full score
