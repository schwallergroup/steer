"""Generated evaluation code for: Late Fischer indole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateIndoleFormation(BaseScoring):
    """
    Evaluates routes based on late-stage indole ring formation via Fischer indole synthesis.
    Rewards routes where the indole ring (c1ccc2[nH]ccc2c1) is formed late in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.indole_smarts = "c1ccc2[nH]ccc2c1"
        self.timing = config.get("timing", "late")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Indole formation doesn't happen
        else:
            # Late-stage formation is better (closer to 0)
            # Convert to 0-10 scale where later formation gets higher score
            return (1 - x) * 10
            
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms an indole ring via Fischer indole synthesis"""
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        # Parse reaction
        reactants_smiles, product_smiles = rxn_smiles.split(">>")
        reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
        product = Chem.MolFromSmiles(product_smiles.strip())
        
        if not product or not all(reactants):
            return False
            
        # Check if product contains indole but reactants don't
        indole_pattern = Chem.MolFromSmarts(self.indole_smarts)
        if not indole_pattern:
            return False
            
        product_has_indole = product.HasSubstructMatch(indole_pattern)
        reactants_have_indole = any(r.HasSubstructMatch(indole_pattern) for r in reactants if r)
        
        # Indole formation: product has indole but reactants don't
        if product_has_indole and not reactants_have_indole:
            # Additional check for Fischer indole characteristics
            return self._is_fischer_indole_synthesis(reactants, product)
            
        return False
        
    def _is_fischer_indole_synthesis(self, reactants, product):
        """Check if reaction pattern matches Fischer indole synthesis"""
        # Fischer indole: aryl hydrazine + ketone/aldehyde -> indole
        hydrazine_pattern = Chem.MolFromSmarts("c-[NH]-[NH2]")  # Aryl hydrazine
        carbonyl_pattern = Chem.MolFromSmarts("[CX3]=[OX1]")     # Ketone/aldehyde
        
        if not hydrazine_pattern or not carbonyl_pattern:
            return False
            
        has_hydrazine = any(r.HasSubstructMatch(hydrazine_pattern) for r in reactants if r)
        has_carbonyl = any(r.HasSubstructMatch(carbonyl_pattern) for r in reactants if r)
        
        return has_hydrazine and has_carbonyl
