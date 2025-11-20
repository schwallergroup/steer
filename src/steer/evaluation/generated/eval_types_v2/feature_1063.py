"""Generated evaluation code for: Late stage amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Evaluates whether amide coupling reactions occur at late stages in the synthesis route.
    
    Detects amide bond formation reactions and rewards routes where such reactions
    happen within the specified depth threshold from the target molecule.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 2)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No amide coupling found
        
        # Convert depth to fraction and reward late-stage reactions
        if x <= self.depth_threshold:
            return 10 * (1 - x)  # Earlier reactions get higher scores
        else:
            return max(0, 10 * (1 - x))  # Penalize very early reactions
    
    def hit_condition(self, d) -> bool:
        """Check if the reaction involves amide bond formation"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Define amide pattern
            amide_pattern = Chem.MolFromSmarts("[C,c](=O)[NH,N]")
            
            # Check if product contains amide bond
            if not product.HasSubstructMatch(amide_pattern):
                return False
            
            # Check if amide bond is formed (not present in reactants)
            product_amide_count = len(product.GetSubstructMatches(amide_pattern))
            reactant_amide_count = sum(len(r.GetSubstructMatches(amide_pattern)) for r in reactants)
            
            # Amide coupling should increase amide count
            if product_amide_count > reactant_amide_count:
                return True
                
            # Additional check for typical amide coupling patterns
            # Look for carboxylic acid/ester + amine patterns in reactants
            carboxyl_pattern = Chem.MolFromSmarts("[C,c](=O)[OH,O]")
            amine_pattern = Chem.MolFromSmarts("[NH2,NH1,N]")
            
            has_carboxyl = any(r.HasSubstructMatch(carboxyl_pattern) for r in reactants)
            has_amine = any(r.HasSubstructMatch(amine_pattern) for r in reactants)
            
            return has_carboxyl and has_amine and product.HasSubstructMatch(amide_pattern)
            
        except Exception:
            return False
