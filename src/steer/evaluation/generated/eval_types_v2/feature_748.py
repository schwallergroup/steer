"""Generated evaluation code for: Late stage Suzuki coupling for biaryl formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiCoupling(BaseScoring):
    """
    Evaluates whether a Suzuki-Miyaura coupling reaction occurs at late stage
    for biaryl formation (C_sp2-C_sp2 bond formation).
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "late")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        else:
            if self.timing_preference == "late":
                return 1 - x  # Later stage is better (lower depth fraction)
            else:
                return x  # Earlier stage is better (higher depth fraction)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction is a Suzuki coupling forming a biaryl bond.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
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
        """
        Detect Suzuki coupling by checking for:
        1. Boronic acid/ester reactant
        2. Aryl halide reactant  
        3. Formation of new C_sp2-C_sp2 bond between aromatic rings
        """
        # Check for boronic acid/ester patterns
        boronic_patterns = [
            "[c,C]=,-[B]([OH])[OH]",  # Boronic acid
            "[c,C]=,-[B]1OC(C)(C)C(C)(C)O1",  # Pinacol ester
            "[c,C]=,-[B](O)(O)"  # Alternative boronic acid
        ]
        
        # Check for aryl halide patterns
        halide_patterns = [
            "[c,C]=,-[Cl,Br,I]"  # Aryl halides
        ]
        
        has_boronic = False
        has_halide = False
        
        for reactant in reactants:
            # Check for boronic acid/ester
            for pattern in boronic_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_boronic = True
                    break
                    
            # Check for aryl halide
            for pattern in halide_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_halide = True
                    break
        
        # Must have both coupling partners
        if not (has_boronic and has_halide):
            return False
            
        # Check if product contains biaryl motif (two connected aromatic rings)
        biaryl_pattern = "c1ccccc1-c2ccccc2"  # Simple biaryl
        heteroaryl_biaryl_patterns = [
            "c1ccc(cc1)-c2ccccn2",  # phenyl-pyridine
            "c1ccc(cc1)-c2ccco2",   # phenyl-furan
            "c1ccc(cc1)-c2ccc[nH]2", # phenyl-pyrrole
            "c1ccc(cc1)-c2cccs2"    # phenyl-thiophene
        ]
        
        # Check for biaryl formation
        if product.HasSubstructMatch(Chem.MolFromSmarts(biaryl_pattern)):
            return True
            
        for pattern in heteroaryl_biaryl_patterns:
            if product.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                return True
                
        return False
