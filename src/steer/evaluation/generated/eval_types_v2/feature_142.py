"""Generated evaluation code for: Late stage selective nitrile hydrolysis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSelectiveNitrileHydrolysis(BaseScoring):
    """
    Evaluates whether a route performs selective nitrile hydrolysis at a late stage.
    Checks for conversion of nitrile to carboxylic acid/amide in presence of other nitriles.
    """
    
    def __init__(self, config: Dict):
        self.timing_threshold = config.get("late_stage_threshold", 0.8)  # Must occur in last 20% of route
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        # Reward very late-stage reactions (close to 1.0 depth fraction)
        if x >= self.timing_threshold:
            return 10 * (x - self.timing_threshold) / (1.0 - self.timing_threshold)
        else:
            return 0  # Too early in the route
            
    def hit_condition(self, d) -> bool:
        """Check if this reaction performs selective nitrile hydrolysis"""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            product = Chem.MolFromSmiles(rxn[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Define patterns
            nitrile_pattern = Chem.MolFromSmarts("[#6]C#N")
            carboxylic_acid_pattern = Chem.MolFromSmarts("[#6]C(=O)[OH]")
            amide_pattern = Chem.MolFromSmarts("[#6]C(=O)[NH2,NH1]")
            
            # Check if we have nitrile hydrolysis (nitrile -> acid or amide)
            has_hydrolysis = False
            
            for reactant in reactants:
                if not reactant.HasSubstructMatch(nitrile_pattern):
                    continue
                    
                # Check if product has corresponding acid or amide
                if (product.HasSubstructMatch(carboxylic_acid_pattern) or 
                    product.HasSubstructMatch(amide_pattern)):
                    
                    # Verify selectivity: check if other nitriles remain unchanged
                    reactant_nitriles = len(reactant.GetSubstructMatches(nitrile_pattern))
                    product_nitriles = len(product.GetSubstructMatches(nitrile_pattern))
                    
                    # Selective if we converted exactly one nitrile and others remain
                    if reactant_nitriles > 1 and product_nitriles == (reactant_nitriles - 1):
                        has_hydrolysis = True
                        break
                        
            return has_hydrolysis
            
        except Exception:
            return False
