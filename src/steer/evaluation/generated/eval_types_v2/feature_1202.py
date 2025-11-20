"""Generated evaluation code for: Selective carboxylic acid reduction over ester"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SelectiveCarboxylicAcidReduction(BaseScoring):
    """
    Evaluates routes for selective carboxylic acid reduction over ester.
    Detects reactions where a carboxylic acid is reduced to alcohol while 
    ester groups remain intact, typically using borane or similar reagents.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("condition_type", "bool")
        self.target_depth = config.get("target_depth", 0)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Selective reduction doesn't happen
        else:
            return 1 - x  # Earlier selective reduction is better
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction performs selective carboxylic acid reduction over ester.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Find the main organic molecule (largest by atom count)
            main_reactant = max(reactants, key=lambda m: m.GetNumAtoms())
            main_product = max(products, key=lambda m: m.GetNumAtoms())
            
            # Check if reactant has both carboxylic acid and ester
            carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")
            ester_pattern = Chem.MolFromSmarts("[C](=O)[O][C]")
            alcohol_pattern = Chem.MolFromSmarts("[CH2][OH]")
            
            reactant_has_carboxylic_acid = main_reactant.HasSubstructMatch(carboxylic_acid_pattern)
            reactant_has_ester = main_reactant.HasSubstructMatch(ester_pattern)
            
            if not (reactant_has_carboxylic_acid and reactant_has_ester):
                return False
            
            # Check if product has alcohol (from reduced carboxylic acid) and still has ester
            product_has_alcohol = main_product.HasSubstructMatch(alcohol_pattern)
            product_has_ester = main_product.HasSubstructMatch(ester_pattern)
            product_has_carboxylic_acid = main_product.HasSubstructMatch(carboxylic_acid_pattern)
            
            # Selective reduction: carboxylic acid -> alcohol, ester remains
            if (product_has_alcohol and 
                product_has_ester and 
                not product_has_carboxylic_acid):
                
                # Additional check: count ester groups to ensure selectivity
                reactant_ester_matches = len(main_reactant.GetSubstructMatches(ester_pattern))
                product_ester_matches = len(main_product.GetSubstructMatches(ester_pattern))
                
                # Ester count should be preserved
                return reactant_ester_matches == product_ester_matches
            
            return False
            
        except Exception:
            return False
