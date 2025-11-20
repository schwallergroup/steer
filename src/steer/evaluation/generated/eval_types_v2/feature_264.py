"""Generated evaluation code for: Ester-nitrile-ester functional group cycling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EsterNitrileEsterCycling(MultiRxnCondBase):
    """
    Detects ester-nitrile-ester functional group cycling in synthesis routes.
    Checks for a sequence of transformations: ester -> acid -> amide -> nitrile -> ester
    """
    
    def __init__(self, config):
        self.reaction_sequence = config["reaction_sequence"]
        self.min_steps = config.get("min_steps", 4)
        
        # Define SMARTS patterns for functional groups
        self.patterns = {
            "ester": "[#6][C](=[O])[O][#6]",
            "acid": "[#6][C](=[O])[OH]", 
            "amide": "[#6][C](=[O])[N]",
            "nitrile": "[#6][C]#[N]"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find the cycling sequence in the reactions
        cycling_found = self.detect_functional_group_cycling(reactions)
        
        return cycling_found, len(reactions)
    
    def detect_functional_group_cycling(self, reactions):
        """Detect if the functional group cycling sequence occurs"""
        if len(reactions) < self.min_steps:
            return False
            
        # Track functional group transformations through the reaction sequence
        transformations = []
        
        for rxn in reactions:
            transformation = self.identify_transformation(rxn)
            if transformation:
                transformations.append(transformation)
        
        # Check if the required sequence appears in transformations
        return self.has_cycling_sequence(transformations)
    
    def identify_transformation(self, rxn):
        """Identify what functional group transformation occurred in a reaction"""
        rxn_smiles = rxn.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles:
            return None
            
        parts = rxn_smiles.split(">>")
        if len(parts) != 2:
            return None
            
        reactants = parts[0]
        products = parts[1].split(".")[0]  # Take first product
        
        try:
            reactant_mol = Chem.MolFromSmiles(reactants.split(".")[0])
            product_mol = Chem.MolFromSmiles(products)
            
            if not reactant_mol or not product_mol:
                return None
            
            # Check what functional groups are present
            reactant_groups = self.get_functional_groups(reactant_mol)
            product_groups = self.get_functional_groups(product_mol)
            
            # Determine transformation type
            for react_fg in reactant_groups:
                for prod_fg in product_groups:
                    if react_fg != prod_fg:
                        return f"{react_fg}_to_{prod_fg}"
            
        except:
            return None
        
        return None
    
    def get_functional_groups(self, mol):
        """Get functional groups present in a molecule"""
        groups = []
        for group_name, pattern in self.patterns.items():
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                groups.append(group_name)
        return groups
    
    def has_cycling_sequence(self, transformations):
        """Check if the transformations contain the required cycling sequence"""
        target_sequence = self.reaction_sequence
        
        # Look for the complete sequence in transformations
        for i in range(len(transformations) - len(target_sequence) + 1):
            sequence_match = True
            for j, expected_transform in enumerate(target_sequence):
                if i + j >= len(transformations) or transformations[i + j] != expected_transform:
                    sequence_match = False
                    break
            
            if sequence_match:
                return True
        
        return False
