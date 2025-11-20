"""Generated evaluation code for: Late stage Buchwald-Hartwig amination"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageBuchwaldHartwig(BaseScoring):
    """
    Evaluates whether a Buchwald-Hartwig amination reaction occurs at a late stage.
    
    Detects C-N bond formation between aryl halides/triflates and amines using
    palladium catalysis. Returns higher scores for reactions occurring later
    in the synthesis (lower depth fraction).
    """
    
    def __init__(self, config: Dict):
        pass
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            return 1 - x  # Later stage (lower x) gets higher score
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Buchwald-Hartwig amination"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            product = Chem.MolFromSmiles(product_smiles.strip())
            
            if not all(reactants) or not product:
                return False
            
            return self._is_buchwald_hartwig(reactants, product)
            
        except Exception:
            return False
    
    def _is_buchwald_hartwig(self, reactants, product):
        """
        Check if reaction involves C-N bond formation between aryl halide and amine
        """
        # Aryl halide patterns (Br, I, Cl on aromatic ring, or triflate)
        aryl_halide_patterns = [
            "[cH0:1][Br:2]",  # Aryl bromide
            "[cH0:1][I:2]",   # Aryl iodide  
            "[cH0:1][Cl:2]",  # Aryl chloride
            "[cH0:1][O:2][S:3](=O)(=O)[C:4](F)(F)F"  # Aryl triflate
        ]
        
        # Amine patterns
        amine_patterns = [
            "[NH2:3]",        # Primary amine
            "[NH:3][CH3]",    # Secondary amine (methyl)
            "[NH:3][c]",      # Aniline-type
            "[NH:3]([CH3])[CH3]"  # Dialkyl amine
        ]
        
        # Product pattern - aryl-nitrogen bond
        product_pattern = "[c:1][NH:3]"
        
        # Check if product contains the expected C-N bond
        product_match = False
        for pattern in ["[c][NH2]", "[c][NH][C]", "[c][NH][c]"]:
            if product.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                product_match = True
                break
        
        if not product_match:
            return False
        
        # Check if reactants contain aryl halide and amine
        has_aryl_halide = False
        has_amine = False
        
        for reactant in reactants:
            # Check for aryl halide
            for pattern in aryl_halide_patterns:
                try:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_aryl_halide = True
                        break
                except:
                    continue
            
            # Check for amine
            for pattern in amine_patterns:
                try:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_amine = True
                        break
                except:
                    continue
        
        return has_aryl_halide and has_amine
