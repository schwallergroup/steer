"""Generated evaluation code for: Late stage amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Evaluates whether amide coupling occurs at a late stage in the synthesis route.
    Rewards routes where amide bond formation happens after the specified stage threshold.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config.get("stage_threshold", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Amide coupling doesn't happen
        elif x >= self.stage_threshold:
            return 10  # Perfect score for very late stage
        else:
            # Linear scaling: later is better
            return 10 * (x / self.stage_threshold)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents amide coupling"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            product_smiles, reactants_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            return self._is_amide_coupling(product, reactants)
            
        except:
            return False
    
    def _is_amide_coupling(self, product, reactants) -> bool:
        """
        Detect amide coupling by checking for:
        1. Amide bond formation in product
        2. Carboxylic acid/ester + amine in reactants
        """
        # Check for amide bond in product
        amide_pattern = Chem.MolFromSmarts("[C](=[O])[NH]")
        if not product.HasSubstructMatch(amide_pattern):
            return False
            
        # Check for carboxylic acid, ester, or acid chloride patterns
        carboxyl_patterns = [
            "[C](=[O])[OH]",  # carboxylic acid
            "[C](=[O])[O][C]",  # ester
            "[C](=[O])[Cl]",  # acid chloride
            "[C](=[O])[F]"   # acid fluoride
        ]
        
        # Check for amine patterns
        amine_patterns = [
            "[NH2]",  # primary amine
            "[NH1]",  # secondary amine
            "[c][NH2]",  # aniline
            "[c][NH1]"   # N-substituted aniline
        ]
        
        has_carboxyl = False
        has_amine = False
        
        for reactant in reactants:
            # Check for carboxyl component
            for pattern_smarts in carboxyl_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if reactant.HasSubstructMatch(pattern):
                    has_carboxyl = True
                    break
                    
            # Check for amine component
            for pattern_smarts in amine_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if reactant.HasSubstructMatch(pattern):
                    has_amine = True
                    break
                    
        return has_carboxyl and has_amine
