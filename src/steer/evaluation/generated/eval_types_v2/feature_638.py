"""Generated evaluation code for: Late stage nitrile hydrolysis to carboxylic acid"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNitrileHydrolysis(BaseScoring):
    """
    Evaluates synthesis routes for late-stage nitrile hydrolysis to carboxylic acid.
    Returns higher scores when nitrile hydrolysis occurs later in the synthesis,
    as this harsh reaction can cause deprotection and epimerization issues.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)  # Default prefer late stage

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Nitrile hydrolysis doesn't occur
        else:
            # Higher score for later stage hydrolysis (closer to 1.0 depth fraction)
            return 10 * x  # Scale 0-1 depth fraction to 0-10 score

    def hit_condition(self, d) -> bool:
        """
        Detects nitrile hydrolysis to carboxylic acid by checking for:
        1. Nitrile group in reactants
        2. Carboxylic acid group in product
        3. Same carbon atom involved in both groups (via atom mapping)
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            product_smiles, reactant_smiles = rxn_smiles.split(">>")
            product_mol = Chem.MolFromSmiles(product_smiles)
            
            if product_mol is None:
                return False
                
            # Check for carboxylic acid in product
            carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=[O])[OH]")
            if not product_mol.HasSubstructMatch(carboxylic_acid_pattern):
                return False
                
            # Get atom map numbers of carboxylic acid carbons in product
            carboxylic_matches = product_mol.GetSubstructMatches(carboxylic_acid_pattern)
            carboxylic_carbons = set()
            for match in carboxylic_matches:
                carbon_atom = product_mol.GetAtomWithIdx(match[0])
                if carbon_atom.GetAtomMapNum() > 0:
                    carboxylic_carbons.add(carbon_atom.GetAtomMapNum())
                    
            if not carboxylic_carbons:
                return False
                
            # Check reactants for nitrile groups on same mapped carbons
            nitrile_pattern = Chem.MolFromSmarts("[C]#[N]")
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            for reactant_mol in reactant_mols:
                if reactant_mol is None:
                    continue
                    
                if reactant_mol.HasSubstructMatch(nitrile_pattern):
                    nitrile_matches = reactant_mol.GetSubstructMatches(nitrile_pattern)
                    for match in nitrile_matches:
                        nitrile_carbon = reactant_mol.GetAtomWithIdx(match[0])
                        if nitrile_carbon.GetAtomMapNum() in carboxylic_carbons:
                            return True
                            
            return False
            
        except Exception:
            return False
