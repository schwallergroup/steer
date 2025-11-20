"""Generated evaluation code for: Convergent synthesis via Suzuki cross-coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SuzukiConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis via Suzuki cross-coupling.
    Detects palladium-catalyzed C-C bond formation between aryl/vinyl halides 
    and boronic acids/esters to form biaryl linkages.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        self.min_fragment_count = config.get("fragment_count", 2)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        else:
            if self.condition_type == "bool":
                return 1  # Condition met
            else:
                # Earlier convergent coupling is generally better
                return max(0, 1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Detects Suzuki cross-coupling by identifying:
        1. Formation of C-C bond between aromatic carbons
        2. Presence of organoborane reactant (B-C bond)
        3. Presence of aryl/vinyl halide reactant
        4. Convergent nature (2+ substantial fragments)
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            prod_smiles, react_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in react_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Check for sufficient fragment count
            if len(reactants) < self.min_fragment_count:
                return False
                
            # Check for organoborane reactant (boronic acid/ester)
            borane_patterns = [
                "[#6]-[#5](-[#8])-[#8]",  # Boronic acid
                "[#6]-[#5]1-[#8]-[#6]-[#6]-[#8]-1",  # Boronic ester (pinacol)
                "[#6]-[#5](-[#8]-[#6])-[#8]-[#6]"  # Simple boronic ester
            ]
            
            has_borane = False
            for reactant in reactants:
                for pattern in borane_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_borane = True
                        break
                if has_borane:
                    break
                    
            if not has_borane:
                return False
                
            # Check for aryl/vinyl halide reactant
            halide_patterns = [
                "[#6]=[#6]-[#9,#17,#35,#53]",  # Vinyl halide
                "c-[#9,#17,#35,#53]",  # Aryl halide
                "[#6](-[#9,#17,#35,#53]):[#6]"  # Aromatic C-X
            ]
            
            has_halide = False
            for reactant in reactants:
                for pattern in halide_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_halide = True
                        break
                if has_halide:
                    break
                    
            if not has_halide:
                return False
                
            # Check for biaryl/conjugated C-C bond formation
            # Look for newly formed C-C bonds between aromatic/sp2 carbons
            biaryl_patterns = [
                "c-c",  # Direct aryl-aryl
                "c-[#6]=[#6]",  # Aryl-vinyl
                "[#6]=[#6]-c",  # Vinyl-aryl
                "[#6]=[#6]-[#6]=[#6]"  # Vinyl-vinyl
            ]
            
            has_cc_formation = False
            for pattern in biaryl_patterns:
                if product.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_cc_formation = True
                    break
                    
            return has_borane and has_halide and has_cc_formation
            
        except Exception:
            return False
