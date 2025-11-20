"""Generated evaluation code for: Late stage Suzuki cross-coupling for biaryl formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzuki(BaseScoring):
    """
    Evaluates whether Suzuki cross-coupling occurs at a late stage in the synthesis route.
    Checks for biaryl formation via Suzuki coupling within the specified number of steps from the end.
    """
    
    def __init__(self, config: Dict):
        self.step_position_from_end = config.get("step_position_from_end", 2)
        self.timing = config.get("timing", "late")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        
        # For late-stage preference, lower depth fraction is better
        if self.timing == "late":
            if x >= 0.8:  # Very late stage (last 20% of steps)
                return 10
            elif x >= 0.6:  # Moderately late stage
                return 7
            elif x >= 0.4:  # Middle stage
                return 4
            else:  # Early stage
                return 1
        else:
            # If not specifically late timing, just reward presence
            return 5
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Suzuki cross-coupling forming biaryl."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for Suzuki coupling indicators
            return self._is_suzuki_coupling(product, reactants)
            
        except Exception:
            return False
    
    def _is_suzuki_coupling(self, product, reactants) -> bool:
        """Detect Suzuki coupling by checking for biaryl formation and typical reactants."""
        
        # Check if product contains biaryl system
        biaryl_patterns = [
            "[cR1]:[cR1]-[cR1]:[cR1]",  # Simple biaryl
            "c1ccccc1-c2ccccc2",        # Biphenyl-like
            "[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2"  # More specific biaryl
        ]
        
        has_biaryl = False
        for pattern in biaryl_patterns:
            if product.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                has_biaryl = True
                break
        
        if not has_biaryl:
            return False
        
        # Check for typical Suzuki reactants: boronic acid/ester and aryl halide
        boronic_patterns = [
            "[#6]-B(-O)-O",      # Boronic acid
            "[#6]-B(O)O",        # Boronic acid alternative
            "[#6]-B1OCCCO1",     # Boronic ester (pinacol)
        ]
        
        halide_patterns = [
            "[cR1]-[Cl,Br,I]",   # Aryl halide
            "[#6]=[#6]-[Cl,Br,I]" # Vinyl halide
        ]
        
        has_boronic = False
        has_halide = False
        
        for reactant in reactants:
            # Check for boronic acid/ester
            for pattern in boronic_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_boronic = True
                    break
            
            # Check for aryl/vinyl halide
            for pattern in halide_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_halide = True
                    break
        
        # Suzuki coupling typically requires both boronic component and halide
        return has_boronic and has_halide
