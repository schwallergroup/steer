"""Generated evaluation code for: Late stage palladium-catalyzed cyanation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStagePdCyanation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage palladium-catalyzed cyanation reactions.
    Detects conversion of aryl halides (Br, I, Cl) to nitriles using palladium catalysis.
    Rewards routes where this transformation occurs at late stages (closer to final product).
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "late")  # "late", "early", or "any"
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cyanation doesn't occur
        
        if self.timing_preference == "late":
            return (1 - x) * 10  # Higher score for later stages
        elif self.timing_preference == "early":
            return x * 10  # Higher score for earlier stages
        else:  # "any"
            return 10  # Just presence matters
    
    def hit_condition(self, d) -> bool:
        """
        Detects palladium-catalyzed cyanation by checking for:
        1. Aryl halide in reactants (Br, I, Cl)
        2. Nitrile formation in product
        3. Same aromatic core maintained
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for nitrile formation in product
            nitrile_pattern = Chem.MolFromSmarts("[#6]#[#7]")  # C≡N
            if not product.HasSubstructMatch(nitrile_pattern):
                return False
            
            # Check for aryl halide consumption in reactants
            aryl_bromide = Chem.MolFromSmarts("c-Br")  # Aromatic C-Br
            aryl_iodide = Chem.MolFromSmarts("c-I")    # Aromatic C-I
            aryl_chloride = Chem.MolFromSmarts("c-Cl") # Aromatic C-Cl
            
            has_aryl_halide = False
            for reactant in reactants:
                if (reactant.HasSubstructMatch(aryl_bromide) or 
                    reactant.HasSubstructMatch(aryl_iodide) or 
                    reactant.HasSubstructMatch(aryl_chloride)):
                    has_aryl_halide = True
                    break
            
            if not has_aryl_halide:
                return False
            
            # Additional check: cyanide source in reactants (CN-, Zn(CN)2, etc.)
            cyanide_sources = [
                Chem.MolFromSmarts("[#6-]#[#7]"),     # CN-
                Chem.MolFromSmarts("[Zn]([#6]#[#7])([#6]#[#7])"),  # Zn(CN)2
                Chem.MolFromSmarts("[K][#6]#[#7]"),   # KCN
                Chem.MolFromSmarts("[Na][#6]#[#7]"),  # NaCN
            ]
            
            has_cyanide_source = False
            for reactant in reactants:
                for pattern in cyanide_sources:
                    if pattern and reactant.HasSubstructMatch(pattern):
                        has_cyanide_source = True
                        break
                if has_cyanide_source:
                    break
            
            return has_aryl_halide and has_cyanide_source
            
        except Exception:
            return False
