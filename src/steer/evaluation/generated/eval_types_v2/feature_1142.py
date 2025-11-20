"""Generated evaluation code for: Late stage amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Evaluates whether amide coupling occurs at the final step of synthesis.
    Returns high score (close to 10) when amide coupling happens as the last reaction,
    lower scores for earlier amide coupling, and 0 if no amide coupling is detected.
    """
    
    def __init__(self, config: Dict):
        self.step_position = config.get("step_position", "final")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No amide coupling found
        
        if self.step_position == "final":
            # For final step, we want x to be close to 1.0 (latest possible)
            if x >= 0.9:  # Very late stage (final step)
                return 10
            elif x >= 0.7:  # Reasonably late
                return 7
            elif x >= 0.5:  # Mid-stage
                return 4
            else:  # Early stage
                return 2
        else:
            # For other timing preferences, adjust scoring accordingly
            return max(0, 10 - abs(x - 0.5) * 10)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves amide coupling"""
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
            
            # Define amide bond pattern
            amide_pattern = Chem.MolFromSmarts("[C](=[O])[NH,N]")
            
            if not amide_pattern:
                return False
            
            # Check if product has more amide bonds than any single reactant
            product_amide_count = len(product.GetSubstructMatches(amide_pattern))
            
            for reactant in reactants:
                reactant_amide_count = len(reactant.GetSubstructMatches(reactant_amide_count))
                
            max_reactant_amides = max([len(r.GetSubstructMatches(amide_pattern)) for r in reactants], default=0)
            
            # Amide coupling if product has more amide bonds than starting materials
            if product_amide_count > max_reactant_amides:
                return True
                
            # Alternative check: look for carboxylic acid + amine pattern in reactants
            carboxylic_acid = Chem.MolFromSmarts("[C](=[O])[OH]")
            amine = Chem.MolFromSmarts("[NH2,NH1]")
            
            has_acid = any(r.HasSubstructMatch(carboxylic_acid) for r in reactants)
            has_amine = any(r.HasSubstructMatch(amine) for r in reactants)
            
            return has_acid and has_amine and product.HasSubstructMatch(amide_pattern)
            
        except Exception:
            return False
