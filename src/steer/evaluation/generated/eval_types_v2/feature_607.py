"""Generated evaluation code for: Late stage amide coupling approach"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Evaluates synthesis routes for late-stage amide coupling reactions.
    Rewards routes where amide bond formation occurs closer to the final product.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)  # Late stage default
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No amide coupling found
        else:
            # Late-stage coupling is better (lower depth fraction)
            # Scale to 0-10 where 10 is best (earliest coupling)
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents an amide coupling."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if an amide bond is formed
            return self._is_amide_coupling_reaction(reactants, product)
            
        except Exception:
            return False
    
    def _is_amide_coupling_reaction(self, reactants, product) -> bool:
        """
        Check if reactants contain carboxylic acid/derivative and amine,
        and product contains newly formed amide bond.
        """
        # Amide pattern (C(=O)N)
        amide_pattern = Chem.MolFromSmarts("[C](=[O])[N]")
        if not amide_pattern:
            return False
            
        # Check if product has amide bond
        if not product.HasSubstructMatch(amide_pattern):
            return False
        
        # Carboxylic acid or activated ester patterns
        carboxylic_patterns = [
            "[C](=[O])[OH]",  # Carboxylic acid
            "[C](=[O])[O][C]",  # Ester
            "[C](=[O])[Cl]",  # Acyl chloride
            "[C](=[O])[O][C](=[O])[C]"  # Anhydride
        ]
        
        # Amine patterns
        amine_patterns = [
            "[N;!$(N=*);!$(N#*);!$([N]([O])=O)]",  # Primary/secondary amine
            "[NH2]",  # Primary amine
            "[NH1]"   # Secondary amine
        ]
        
        has_carboxylic = False
        has_amine = False
        
        for reactant in reactants:
            if not has_carboxylic:
                for pattern_smarts in carboxylic_patterns:
                    pattern = Chem.MolFromSmarts(pattern_smarts)
                    if pattern and reactant.HasSubstructMatch(pattern):
                        has_carboxylic = True
                        break
            
            if not has_amine:
                for pattern_smarts in amine_patterns:
                    pattern = Chem.MolFromSmarts(pattern_smarts)
                    if pattern and reactant.HasSubstructMatch(pattern):
                        has_amine = True
                        break
        
        return has_carboxylic and has_amine
