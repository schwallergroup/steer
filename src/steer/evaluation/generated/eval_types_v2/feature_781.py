"""Generated evaluation code for: Linear amine to ester functional group progression"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LinearAmineToEsterProgression(MultiRxnCondBase):
    """
    Evaluates whether a synthesis route follows a linear progression from amine to ester
    through the specific sequence: sandmeyer -> cyanation -> grignard_nitrile -> haloform -> esterification
    """
    
    def __init__(self, config):
        self.required_sequence = config["parameters"]["sequence"]
        self.require_linearity = config["parameters"]["linearity"]
        
        # Define reaction patterns for each transformation
        self.reaction_patterns = {
            "sandmeyer": {
                "reactant": "c[NH2]",  # aromatic amine
                "product": "c[Cl,Br,I]"  # aromatic halide
            },
            "cyanation": {
                "reactant": "c[Cl,Br,I]",  # aromatic halide
                "product": "c[C]#[N]"  # aromatic nitrile
            },
            "grignard_nitrile": {
                "reactant": "c[C]#[N]",  # aromatic nitrile
                "product": "c[C](=[O])[CH3]"  # aromatic methyl ketone
            },
            "haloform": {
                "reactant": "c[C](=[O])[CH3]",  # aromatic methyl ketone
                "product": "c[C](=[O])[OH]"  # aromatic carboxylic acid
            },
            "esterification": {
                "reactant": "c[C](=[O])[OH]",  # aromatic carboxylic acid
                "product": "c[C](=[O])[O][C]"  # aromatic ester
            }
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Identify which reactions match each transformation type
        reaction_types = []
        for rxn in reactions:
            rxn_type = self.identify_reaction_type(rxn)
            if rxn_type:
                reaction_types.append(rxn_type)
        
        # Check if the sequence matches our required progression
        sequence_match = self.check_sequence_match(reaction_types)
        
        # If linearity is required, check that reactions follow exact order
        if self.require_linearity:
            condition = sequence_match and self.is_linear_sequence(reaction_types)
        else:
            condition = sequence_match
            
        return condition, len(reactions)
    
    def identify_reaction_type(self, rxn):
        """Identify which transformation type a reaction represents"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return None
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            reactant_mols = [Chem.MolFromSmiles(s) for s in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(s) for s in products_smiles.split(".")]
            
            if not all(reactant_mols + product_mols):
                return None
                
        except:
            return None
        
        # Check each reaction type pattern
        for rxn_type, patterns in self.reaction_patterns.items():
            reactant_pattern = Chem.MolFromSmarts(patterns["reactant"])
            product_pattern = Chem.MolFromSmarts(patterns["product"])
            
            if not (reactant_pattern and product_pattern):
                continue
                
            # Check if reactants contain the expected starting material
            reactant_match = any(mol.HasSubstructMatch(reactant_pattern) for mol in reactant_mols)
            
            # Check if products contain the expected product
            product_match = any(mol.HasSubstructMatch(product_pattern) for mol in product_mols)
            
            if reactant_match and product_match:
                return rxn_type
                
        return None
    
    def check_sequence_match(self, reaction_types):
        """Check if all required reaction types are present"""
        required_set = set(self.required_sequence)
        found_set = set(reaction_types)
        return required_set.issubset(found_set)
    
    def is_linear_sequence(self, reaction_types):
        """Check if reactions follow the exact linear order"""
        if len(reaction_types) < len(self.required_sequence):
            return False
            
        # Find the starting position of our sequence in the reaction list
        for i in range(len(reaction_types) - len(self.required_sequence) + 1):
            subsequence = reaction_types[i:i + len(self.required_sequence)]
            if subsequence == self.required_sequence:
                return True
                
        return False
