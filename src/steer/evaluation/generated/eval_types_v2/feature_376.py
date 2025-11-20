"""Generated evaluation code for: Late stage ether formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageEtherFormation(BaseScoring):
    """
    Evaluates whether Williamson ether synthesis occurs at a late stage in the route.
    Late stage is defined as occurring within the final 20% of the synthesis depth.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config.get("stage_threshold", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ether formation doesn't happen
        elif x >= self.stage_threshold:
            return 10  # Perfect score for very late stage
        else:
            # Linear scoring: later is better
            return 10 * (x / self.stage_threshold)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Williamson ether synthesis"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            prod_smiles, react_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            return self._is_williamson_ether_synthesis(product, reactants)
            
        except Exception:
            return False
    
    def _is_williamson_ether_synthesis(self, product, reactants):
        """
        Detect Williamson ether synthesis by checking:
        1. Product has an ether linkage (C-O-C)
        2. Reactants contain alkyl halide/tosylate and alkoxide/alcohol patterns
        """
        # Check if product contains ether group
        ether_pattern = Chem.MolFromSmarts("[C]-[O]-[C]")
        if not product.HasSubstructMatch(ether_pattern):
            return False
            
        # Look for characteristic reactant patterns
        alkyl_halide_pattern = Chem.MolFromSmarts("[C][Cl,Br,I]")  # Alkyl halide
        tosylate_pattern = Chem.MolFromSmarts("[C]OS(=O)(=O)c1ccc(C)cc1")  # Tosylate
        alcohol_pattern = Chem.MolFromSmarts("[C][OH]")  # Alcohol
        alkoxide_pattern = Chem.MolFromSmarts("[C][O-]")  # Alkoxide
        
        has_leaving_group = False
        has_nucleophile = False
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(alkyl_halide_pattern) or \
               reactant.HasSubstructMatch(tosylate_pattern):
                has_leaving_group = True
            elif reactant.HasSubstructMatch(alcohol_pattern) or \
                 reactant.HasSubstructMatch(alkoxide_pattern):
                has_nucleophile = True
                
        return has_leaving_group and has_nucleophile
