"""Generated evaluation code for: Late thiazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateThiazoleFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage thiazole ring formation using Hantzsch synthesis.
    Checks for the formation of thiazole rings (c1scnc1) and rewards routes where this 
    formation occurs later in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.formation_method = config["parameters"]["formation_method"]
        self.thiazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No thiazole formation found
        else:
            # Later formation is better for late timing preference
            return 1 - x if self.timing == "late" else x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves thiazole ring formation via Hantzsch synthesis.
        """
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles:
            return False
            
        try:
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains thiazole ring
            product_has_thiazole = product.HasSubstructMatch(self.thiazole_pattern)
            
            # Check if any reactant contains thiazole ring
            reactants_have_thiazole = any(r.HasSubstructMatch(self.thiazole_pattern) for r in reactants)
            
            # Thiazole formation: product has thiazole but reactants don't
            if product_has_thiazole and not reactants_have_thiazole:
                # Additional check for Hantzsch-like pattern if specified
                if self.formation_method == "hantzsch":
                    return self._is_hantzsch_like_reaction(reactants, product)
                return True
                
        except Exception:
            return False
            
        return False
    
    def _is_hantzsch_like_reaction(self, reactants, product) -> bool:
        """
        Check if the reaction resembles Hantzsch thiazole synthesis pattern.
        Typical Hantzsch involves alpha-haloketone + thioamide.
        """
        # Look for patterns typical in Hantzsch synthesis
        haloketone_pattern = Chem.MolFromSmarts("[CH2,CH][Cl,Br,I]")  # Alpha-haloketone-like
        thioamide_pattern = Chem.MolFromSmarts("C(=S)N")  # Thioamide
        carbonyl_pattern = Chem.MolFromSmarts("C=O")  # General carbonyl
        sulfur_pattern = Chem.MolFromSmarts("[S]")  # Sulfur source
        
        has_haloketone = any(r.HasSubstructMatch(haloketone_pattern) for r in reactants)
        has_thioamide = any(r.HasSubstructMatch(thioamide_pattern) for r in reactants)
        has_carbonyl = any(r.HasSubstructMatch(carbonyl_pattern) for r in reactants)
        has_sulfur = any(r.HasSubstructMatch(sulfur_pattern) for r in reactants)
        
        # Flexible matching - either classic thioamide or carbonyl + sulfur source
        return (has_haloketone or has_carbonyl) and (has_thioamide or has_sulfur)
