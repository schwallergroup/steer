"""Generated evaluation code for: Late stage amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Evaluates whether amide coupling occurs at the final step (position 1) of the synthesis route.
    Returns higher scores when amide bond formation happens late in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.target_position = config["parameters"].get("position", 1)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Amide coupling doesn't happen
        else:
            # Late-stage amide coupling is better, perfect score at final step
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents an amide coupling reaction"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            product_smiles, reactants_smiles = rxn_smiles.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            return self._is_amide_coupling(product, reactants)
            
        except Exception:
            return False
    
    def _is_amide_coupling(self, product, reactants) -> bool:
        """
        Detect amide coupling by checking if:
        1. Product contains amide bond that wasn't in reactants
        2. Reactants contain carboxylic acid/ester and amine patterns
        """
        # Amide bond pattern
        amide_pattern = Chem.MolFromSmarts("[C](=[O])[N]")
        if not amide_pattern:
            return False
            
        # Check if product has amide bonds
        product_amides = product.GetSubstructMatches(amide_pattern)
        if not product_amides:
            return False
            
        # Check if any reactant already has the same amide bonds
        reactant_amide_count = sum(len(r.GetSubstructMatches(amide_pattern)) for r in reactants)
        product_amide_count = len(product_amides)
        
        # New amide bond formed
        if product_amide_count <= reactant_amide_count:
            return False
            
        # Look for carboxylic acid/ester and amine in reactants
        carboxylic_pattern = Chem.MolFromSmarts("[C](=[O])[OH,O]")
        amine_pattern = Chem.MolFromSmarts("[N;!$(N=*);!$(N#*)]")
        
        has_carboxylic = any(r.HasSubstructMatch(carboxylic_pattern) for r in reactants)
        has_amine = any(r.HasSubstructMatch(amine_pattern) for r in reactants)
        
        return has_carboxylic and has_amine
