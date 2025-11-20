"""Generated evaluation code for: Early azide introduction via SN2 displacement"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyAzideIntroduction(BaseScoring):
    """
    Evaluates if azide introduction occurs early via SN2 displacement of mesylate.
    Rewards early-stage azide introduction through nucleophilic substitution.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)  # Early in route
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Reward early occurrence (lower depth fraction is better)
            if self.condition_type == "bool":
                return 1 if x <= self.target_depth else 0
            else:
                if x <= self.target_depth:
                    return 1.0
                else:
                    # Penalize late azide introduction
                    return max(0, 1.0 - (x - self.target_depth) * 2)
    
    def hit_condition(self, d):
        """
        Check if this reaction involves azide SN2 displacement of mesylate
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles:
                return False
            
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for azide introduction (N=[N+]=[N-] or [N-][N+]#N)
            azide_patterns = [
                Chem.MolFromSmarts("[N:1]=[N+:2]=[N-:3]"),  # Neutral azide
                Chem.MolFromSmarts("[N-:1][N+:2]#[N:3]"),   # Ionic azide
                Chem.MolFromSmarts("N=[N+]=[N-]"),          # Simple azide
                Chem.MolFromSmarts("[N-][N+]#N")            # Simple ionic azide
            ]
            
            # Check if product contains azide
            has_azide_product = any(product.HasSubstructMatch(pattern) for pattern in azide_patterns if pattern)
            if not has_azide_product:
                return False
            
            # Check for azide reagent in reactants
            has_azide_reagent = False
            for reactant in reactants:
                if any(reactant.HasSubstructMatch(pattern) for pattern in azide_patterns if pattern):
                    has_azide_reagent = True
                    break
            
            if not has_azide_reagent:
                return False
            
            # Check for mesylate leaving group patterns
            mesylate_patterns = [
                Chem.MolFromSmarts("COS(=O)(=O)C"),  # Mesylate ester
                Chem.MolFromSmarts("[CH2,CH:1]OS(=O)(=O)C"),  # Carbon attached to mesylate
            ]
            
            has_mesylate = False
            for reactant in reactants:
                if any(reactant.HasSubstructMatch(pattern) for pattern in mesylate_patterns if pattern):
                    has_mesylate = True
                    break
            
            # Alternative: check for displacement pattern by comparing reactant/product
            if not has_mesylate:
                # Look for carbon that loses mesylate and gains azide
                for reactant in reactants:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts("COS(=O)(=O)C")):
                        has_mesylate = True
                        break
            
            return has_azide_reagent and has_mesylate
            
        except Exception:
            return False
