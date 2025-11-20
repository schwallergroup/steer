"""Generated evaluation code for: Late purine ring formation via annulation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePurineRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage purine ring formation via annulation.
    Checks if a purine ring system is formed through cyclization in the later stages
    of the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.formation_method = config["parameters"]["formation_method"]
        self.purine_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # For late-stage formation, lower depth fraction is better
            # Convert to 0-10 scale where later formation gets higher score
            if self.timing == "late":
                return (1 - x) * 10
            else:
                return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves purine ring formation via annulation.
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
                
            # Check if product contains purine ring
            if not product.HasSubstructMatch(self.purine_pattern):
                return False
                
            # Check if any reactant already contains the complete purine ring
            # If so, this is not a ring formation step
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.purine_pattern):
                    return False
                    
            # Check for annulation pattern: at least one reactant should contain
            # a partial ring system that gets completed
            partial_purine_patterns = [
                "[#7]1[#6][#7][#6]2[#7][#6][#6][#7H][#6]12",  # Adenine precursor
                "[#7]1[#6][#7][#6]2[#7][#6]([#8])[#7][#6]12",  # Guanine precursor
                "[#6]1[#7][#6][#7][#6]2[#7][#6][#7][#6]12"     # Alternative pattern
            ]
            
            has_partial_ring = False
            for reactant in reactants:
                for pattern_smarts in partial_purine_patterns:
                    partial_pattern = Chem.MolFromSmarts(pattern_smarts)
                    if partial_pattern and reactant.HasSubstructMatch(partial_pattern):
                        has_partial_ring = True
                        break
                if has_partial_ring:
                    break
                    
            # Also check for simpler precursors that could form purine via annulation
            if not has_partial_ring:
                # Look for pyrimidine or imidazole rings that could be annulated
                pyrimidine_pattern = Chem.MolFromSmarts("[#7]1[#6][#7][#6][#6][#6]1")
                imidazole_pattern = Chem.MolFromSmarts("[#7]1[#6][#7][#6][#6]1")
                
                for reactant in reactants:
                    if (reactant.HasSubstructMatch(pyrimidine_pattern) or 
                        reactant.HasSubstructMatch(imidazole_pattern)):
                        has_partial_ring = True
                        break
                        
            return has_partial_ring
            
        except Exception:
            return False
