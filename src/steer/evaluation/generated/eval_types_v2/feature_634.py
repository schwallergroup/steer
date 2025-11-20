"""Generated evaluation code for: Late stage nitrile hydrolysis to carboxylic acid"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNitrileHydrolysis(BaseScoring):
    """
    Evaluates whether nitrile hydrolysis to carboxylic acid occurs at a late stage.
    Returns higher scores for reactions happening closer to the target molecule.
    """
    
    def __init__(self, config: Dict):
        # No additional configuration needed for this feature
        pass
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Nitrile hydrolysis doesn't occur
        else:
            return 1 - x  # Late-stage hydrolysis gets higher score
    
    def hit_condition(self, d) -> bool:
        """
        Check if the reaction involves nitrile hydrolysis to carboxylic acid.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants_smiles = rxn[0]
            products_smiles = rxn[1]
            
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p) for p in products_smiles.split(".")]
            
            # SMARTS patterns for nitrile and carboxylic acid
            nitrile_pattern = Chem.MolFromSmarts("[C]#[N]")
            carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=[O])[OH]")
            
            # Check if reactants contain nitrile
            has_nitrile_reactant = any(
                mol.HasSubstructMatch(nitrile_pattern) 
                for mol in reactants if mol is not None
            )
            
            # Check if products contain carboxylic acid
            has_carboxylic_product = any(
                mol.HasSubstructMatch(carboxylic_acid_pattern) 
                for mol in products if mol is not None
            )
            
            # Check if nitrile is consumed (present in reactants but not products)
            has_nitrile_product = any(
                mol.HasSubstructMatch(nitrile_pattern) 
                for mol in products if mol is not None
            )
            
            # Nitrile hydrolysis: nitrile in reactants, carboxylic acid in products,
            # and nitrile is consumed (not present in products)
            return (has_nitrile_reactant and 
                    has_carboxylic_product and 
                    not has_nitrile_product)
            
        except Exception:
            return False
