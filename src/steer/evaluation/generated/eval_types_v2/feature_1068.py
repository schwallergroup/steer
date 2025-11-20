"""Generated evaluation code for: Late stage amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Evaluates whether amide coupling occurs at late stage (depth <= depth_threshold).
    Returns higher scores when amide bond formation happens closer to the final step.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config["parameters"].get("depth_threshold", 2)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No amide coupling found
        
        # Convert depth to score where lower depth (later stage) gets higher score
        if x <= self.depth_threshold:
            # Perfect score for very late stage (depth 0-2)
            return 10 - (x * 2)  # depth 0->10, depth 1->8, depth 2->6
        else:
            # Decreasing score for earlier stages
            return max(0, 6 - (x - self.depth_threshold))
    
    def hit_condition(self, d):
        """Check if this reaction node represents an amide coupling reaction."""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Look for amide bond formation patterns
            return self._is_amide_coupling(product, reactants)
            
        except (KeyError, AttributeError):
            return False
    
    def _is_amide_coupling(self, product, reactants):
        """Detect if this is an amide coupling reaction."""
        # Amide pattern: C(=O)N
        amide_pattern = Chem.MolFromSmarts("[C](=[O])[N]")
        
        if not product.HasSubstructMatch(amide_pattern):
            return False
        
        # Check if amide bond is newly formed (not present in reactants)
        product_amide_count = len(product.GetSubstructMatches(amide_pattern))
        reactant_amide_count = sum(len(r.GetSubstructMatches(amide_pattern)) for r in reactants)
        
        if product_amide_count <= reactant_amide_count:
            return False
        
        # Look for typical amide coupling patterns in reactants
        # Carboxylic acid or acid derivative
        acid_patterns = [
            "[C](=[O])[OH]",  # Carboxylic acid
            "[C](=[O])[Cl]",  # Acyl chloride
            "[C](=[O])[O][C](=[O])",  # Anhydride
        ]
        
        # Amine pattern
        amine_patterns = [
            "[N;H2,H1]",  # Primary or secondary amine
        ]
        
        has_acid = any(any(r.HasSubstructMatch(Chem.MolFromSmarts(pat)) for pat in acid_patterns) for r in reactants)
        has_amine = any(any(r.HasSubstructMatch(Chem.MolFromSmarts(pat)) for pat in amine_patterns) for r in reactants)
        
        return has_acid and has_amine
