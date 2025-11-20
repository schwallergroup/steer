"""Generated evaluation code for: Late stage Buchwald-Hartwig C-N coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BuchwaldHartwigCNLateStage(BaseScoring):
    """
    Evaluates late-stage Buchwald-Hartwig C-N coupling reactions.
    Detects the formation of C-N bonds via palladium-catalyzed cross-coupling
    between aryl halides/triflates and amines/anilines.
    """
    
    def __init__(self, config: Dict):
        self.stage_preference = config.get("stage", "late")  # "early", "late", or "any"
        
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score, favoring late-stage reactions"""
        if x < 0:
            return 0  # Reaction not found
        
        if self.stage_preference == "late":
            return (1 - x) * 10  # Later is better (higher score)
        elif self.stage_preference == "early":
            return x * 10  # Earlier is better
        else:  # "any"
            return 10  # Just presence matters
    
    def hit_condition(self, d) -> bool:
        """Check if reaction is a Buchwald-Hartwig C-N coupling"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for Buchwald-Hartwig pattern
            return self._is_buchwald_hartwig_coupling(reactants, products)
            
        except Exception:
            return False
    
    def _is_buchwald_hartwig_coupling(self, reactants, products) -> bool:
        """Detect Buchwald-Hartwig C-N coupling pattern"""
        
        # Look for aryl halide/triflate pattern in reactants
        aryl_halide_patterns = [
            "[cH0:1][Cl,Br,I]",  # Aryl chloride/bromide/iodide
            "[cH0:1]OS(=O)(=O)CF3",  # Aryl triflate
            "[cH0:1]OS(=O)(=O)[CH3]",  # Aryl mesylate
        ]
        
        # Look for amine/aniline pattern in reactants
        amine_patterns = [
            "[NX3H2:2]",  # Primary amine
            "[NX3H1:2]",  # Secondary amine
            "[cH0][NX3H2:2]",  # Aniline
            "[cH0][NX3H1:2]",  # N-substituted aniline
        ]
        
        # Look for C-N bond formation in products
        cn_bond_patterns = [
            "[cH0:1][NX3:2]",  # Aryl-nitrogen bond
        ]
        
        # Check if we have aryl halide + amine reactants
        has_aryl_halide = False
        has_amine = False
        
        for reactant in reactants:
            if not has_aryl_halide:
                for pattern in aryl_halide_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_aryl_halide = True
                        break
            
            if not has_amine:
                for pattern in amine_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_amine = True
                        break
        
        if not (has_aryl_halide and has_amine):
            return False
        
        # Check if products contain the expected C-N bond
        has_cn_product = False
        for product in products:
            for pattern in cn_bond_patterns:
                if product.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_cn_product = True
                    break
            if has_cn_product:
                break
        
        return has_cn_product
