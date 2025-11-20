"""Generated evaluation code for: Sequential ester protection cycling approach"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialEsterProtectionCycling(MultiRxnCondBase):
    """
    Checks if a route employs sequential ester protection cycling approach,
    specifically looking for carboxylic acid -> methyl ester -> ethyl ester -> carboxylic acid sequence.
    """
    
    def __init__(self, config):
        self.protection_sequence = config.get("protection_sequence", [
            "carboxylic_acid_to_methyl_ester",
            "methyl_ester_to_ethyl_ester", 
            "ethyl_ester_to_carboxylic_acid"
        ])
        self.functional_group = config.get("functional_group", "carboxylic_acid")
        
        # SMARTS patterns for detection
        self.carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")
        self.methyl_ester_pattern = Chem.MolFromSmarts("[C](=O)[O][CH3]")
        self.ethyl_ester_pattern = Chem.MolFromSmarts("[C](=O)[O][CH2][CH3]")
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track the sequence of transformations
        sequence_found = self.detect_protection_sequence(reactions)
        
        return sequence_found, len(reactions)
    
    def detect_protection_sequence(self, reactions) -> bool:
        """
        Detect if the protection sequence occurs in the reactions.
        Returns True if carboxylic acid -> methyl ester -> ethyl ester -> carboxylic acid is found.
        """
        transformations = []
        
        for rxn in reactions:
            transformation = self.classify_transformation(rxn)
            if transformation:
                transformations.append(transformation)
        
        # Check if the required sequence appears in the transformations
        sequence_str = "->".join(self.protection_sequence)
        transformations_str = "->".join(transformations)
        
        return sequence_str in transformations_str
    
    def classify_transformation(self, rxn) -> str:
        """
        Classify the type of transformation based on reactants and products.
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return None
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse molecules
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split('.')]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split('.')]
            
            if not all(reactant_mols) or not all(product_mols):
                return None
            
            # Check for functional group transformations
            reactant_has_carboxylic = any(mol.HasSubstructMatch(self.carboxylic_acid_pattern) for mol in reactant_mols)
            reactant_has_methyl_ester = any(mol.HasSubstructMatch(self.methyl_ester_pattern) for mol in reactant_mols)
            reactant_has_ethyl_ester = any(mol.HasSubstructMatch(self.ethyl_ester_pattern) for mol in reactant_mols)
            
            product_has_carboxylic = any(mol.HasSubstructMatch(self.carboxylic_acid_pattern) for mol in product_mols)
            product_has_methyl_ester = any(mol.HasSubstructMatch(self.methyl_ester_pattern) for mol in product_mols)
            product_has_ethyl_ester = any(mol.HasSubstructMatch(self.ethyl_ester_pattern) for mol in product_mols)
            
            # Classify transformation
            if reactant_has_carboxylic and product_has_methyl_ester:
                return "carboxylic_acid_to_methyl_ester"
            elif reactant_has_methyl_ester and product_has_ethyl_ester:
                return "methyl_ester_to_ethyl_ester"
            elif reactant_has_ethyl_ester and product_has_carboxylic:
                return "ethyl_ester_to_carboxylic_acid"
            elif reactant_has_carboxylic and product_has_ethyl_ester:
                return "carboxylic_acid_to_ethyl_ester"
            elif reactant_has_methyl_ester and product_has_carboxylic:
                return "methyl_ester_to_carboxylic_acid"
            elif reactant_has_ethyl_ester and product_has_methyl_ester:
                return "ethyl_ester_to_methyl_ester"
                
        except Exception:
            return None
        
        return None
    
    def route_scoring(self, x):
        """
        Score based on whether the protection cycling sequence is present.
        """
        if x < 0:
            return 0  # Sequence not found
        else:
            return 1 - x  # Earlier occurrence is better
