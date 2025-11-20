"""Generated evaluation code for: Late stage amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Evaluates whether amide coupling occurs as the final step in the synthesis route.
    Returns a high score (close to 1) if amide coupling happens in the final step,
    and lower scores for earlier occurrences or absence.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("parameters", {}).get("timing", "final_step")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Amide coupling doesn't happen
        elif x == 0:
            return 1  # Final step - ideal case
        else:
            # Earlier steps get lower scores
            return max(0, 1 - x * 0.3)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves amide coupling"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, product_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            return self._is_amide_coupling(reactants, product)
            
        except Exception:
            return False
    
    def _is_amide_coupling(self, reactants, product) -> bool:
        """Detect if reaction involves amide bond formation"""
        # SMARTS pattern for amide bond: C(=O)N
        amide_pattern = Chem.MolFromSmarts("C(=O)N")
        
        if not amide_pattern:
            return False
        
        # Check if product has more amide bonds than reactants
        product_amides = len(product.GetSubstructMatches(amide_pattern))
        reactant_amides = sum(len(r.GetSubstructMatches(amide_pattern)) for r in reactants)
        
        if product_amides <= reactant_amides:
            return False
        
        # Additional check: look for carboxylic acid/ester + amine pattern
        carboxylic_pattern = Chem.MolFromSmarts("C(=O)[OH,OR1]")  # Carboxylic acid or ester
        amine_pattern = Chem.MolFromSmarts("[NH2,NH1,NH0]")  # Primary, secondary, or tertiary amine
        
        has_carboxylic = any(r.HasSubstructMatch(carboxylic_pattern) for r in reactants)
        has_amine = any(r.HasSubstructMatch(amine_pattern) for r in reactants)
        
        return has_carboxylic and has_amine
