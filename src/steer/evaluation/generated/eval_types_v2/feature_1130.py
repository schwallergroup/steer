"""Generated evaluation code for: Late stage Suzuki coupling for biaryl formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzuki(BaseScoring):
    """
    Evaluates whether a Suzuki coupling reaction occurs in the final step to form a biaryl bond.
    Returns higher scores when Suzuki coupling happens as the last reaction in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        # For final step timing, we want depth 0 (last reaction)
        self.target_depth = 0
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        elif x == 0:
            return 1  # Perfect - happens in final step
        else:
            # Penalize earlier occurrence, but still give some credit
            return max(0, 1 - x * 0.3)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction is a Suzuki coupling that forms a biaryl bond.
        """
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            product_smiles, reactant_smiles = rxn_smiles.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if this is a Suzuki coupling by looking for characteristic patterns
            has_suzuki_pattern = self._is_suzuki_coupling(product, reactants)
            
            # Check if a biaryl bond is formed
            has_biaryl_formation = self._forms_biaryl_bond(product, reactants)
            
            return has_suzuki_pattern and has_biaryl_formation
            
        except Exception:
            return False
    
    def _is_suzuki_coupling(self, product, reactants) -> bool:
        """
        Detect Suzuki coupling by looking for boronic acid/ester and halide patterns.
        """
        # Boronic acid/ester patterns
        boronic_patterns = [
            "[#6][B]([OH])[OH]",  # Boronic acid
            "[#6][B]1OC(C)(C)C(C)(C)O1",  # Pinacol boronate
            "[#6][B]([O])[O]"  # Generic boronate
        ]
        
        # Halide patterns (typically Br, I for Suzuki)
        halide_patterns = [
            "[#6][Br]",  # Aryl bromide
            "[#6][I]"    # Aryl iodide
        ]
        
        has_boronic = False
        has_halide = False
        
        for reactant in reactants:
            # Check for boronic acid/ester
            for pattern in boronic_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_boronic = True
                    break
            
            # Check for halide
            for pattern in halide_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_halide = True
                    break
        
        return has_boronic and has_halide
    
    def _forms_biaryl_bond(self, product, reactants) -> bool:
        """
        Check if a biaryl C-C bond is formed by comparing aromatic systems.
        """
        # Biaryl pattern - two aromatic rings connected by single bond
        biaryl_patterns = [
            "c1ccccc1-c2ccccc2",  # Simple biphenyl
            "[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2",  # Generic biaryl
            "c1:[c,n]:[c,n]:[c,n]:[c,n]:1-c2:[c,n]:[c,n]:[c,n]:[c,n]:2"  # Heteroaromatic biaryl
        ]
        
        # Check if product contains biaryl
        product_has_biaryl = False
        for pattern in biaryl_patterns:
            try:
                if product.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    product_has_biaryl = True
                    break
            except:
                continue
        
        if not product_has_biaryl:
            return False
        
        # Check that reactants don't already have this biaryl system
        for reactant in reactants:
            for pattern in biaryl_patterns:
                try:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        return False  # Biaryl already exists in reactants
                except:
                    continue
        
        return True
