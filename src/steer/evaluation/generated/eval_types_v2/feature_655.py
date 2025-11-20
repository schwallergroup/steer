"""Generated evaluation code for: Late stage aminocarbonylation amide formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAminocarbonylation(BaseScoring):
    """
    Evaluates whether aminocarbonylation amide formation occurs at late stage in synthesis.
    Aminocarbonylation typically involves palladium-catalyzed coupling of aryl halides with 
    amines in the presence of CO to form amides.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config.get("stage_threshold", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Aminocarbonylation doesn't occur
        
        # For late-stage preference, reward reactions that occur after the threshold
        if x >= self.stage_threshold:
            return 10.0  # Perfect score for very late stage
        else:
            # Linear penalty for earlier occurrence
            return max(0, 10.0 * (x / self.stage_threshold))
    
    def hit_condition(self, d) -> bool:
        """
        Detect aminocarbonylation by checking for:
        1. Formation of amide bond (C(=O)N pattern)
        2. Presence of aryl halide reactant
        3. Amine reactant
        4. Loss of halogen in product
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, product_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for amide formation in product
            amide_pattern = Chem.MolFromSmarts("[C](=[O])-[N]")
            if not product.HasSubstructMatch(amide_pattern):
                return False
            
            # Check for aryl halide reactant
            aryl_halide_patterns = [
                Chem.MolFromSmarts("c-[Cl,Br,I]"),  # Aryl halides
                Chem.MolFromSmarts("[c,C]=[c,C]-[Cl,Br,I]")  # Vinyl halides
            ]
            
            has_halide_reactant = False
            for reactant in reactants:
                if any(reactant.HasSubstructMatch(pattern) for pattern in aryl_halide_patterns):
                    has_halide_reactant = True
                    break
            
            if not has_halide_reactant:
                return False
            
            # Check for amine reactant (primary or secondary)
            amine_patterns = [
                Chem.MolFromSmarts("[N;H2,H1]"),  # Primary or secondary amine
                Chem.MolFromSmarts("[N;H2]-[C]"),  # Primary amine
                Chem.MolFromSmarts("[N;H1](-[C])-[C]")  # Secondary amine
            ]
            
            has_amine_reactant = False
            for reactant in reactants:
                if any(reactant.HasSubstructMatch(pattern) for pattern in amine_patterns):
                    has_amine_reactant = True
                    break
            
            if not has_amine_reactant:
                return False
            
            # Verify halogen is consumed (present in reactants but not product)
            reactant_halogens = sum(len(r.GetSubstructMatches(Chem.MolFromSmarts("[Cl,Br,I]"))) 
                                  for r in reactants)
            product_halogens = len(product.GetSubstructMatches(Chem.MolFromSmarts("[Cl,Br,I]")))
            
            return reactant_halogens > product_halogens
            
        except Exception:
            return False
