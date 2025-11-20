"""Generated evaluation code for: Late stage Suzuki coupling for biaryl formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiCoupling(BaseScoring):
    """
    Evaluates if a Suzuki-Miyaura coupling reaction occurs at a late stage in the synthesis.
    Specifically looks for C(sp2)-C(sp2) bond formation between aromatic rings.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.0)  # Final step
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        else:
            # Late-stage (close to 0) is better, penalize early reactions
            if self.condition_type == "bool":
                return 1 if x <= 0.2 else 0  # Within 20% of final steps
            else:
                return max(0, 1 - x * 5)  # Linear penalty for early occurrence
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents a Suzuki coupling"""
        metadata = d.get("metadata", {})
        
        # Check if mapped reaction SMILES is available
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for Suzuki coupling characteristics:
            # 1. Boronic acid/ester pattern in reactants
            # 2. Halide pattern in reactants  
            # 3. New C(sp2)-C(sp2) bond in product
            
            has_boron_reactant = self._has_boronic_component(reactants)
            has_halide_reactant = self._has_aryl_halide(reactants)
            forms_biaryl = self._forms_new_biaryl_bond(product, reactants)
            
            return has_boron_reactant and has_halide_reactant and forms_biaryl
            
        except Exception:
            return False
    
    def _has_boronic_component(self, reactants) -> bool:
        """Check for boronic acid or ester patterns"""
        boronic_acid = Chem.MolFromSmarts("[#6]-B(-O)-O")
        boronic_ester = Chem.MolFromSmarts("[#6]-B1-O-C-C-O-1")  # Pinacol ester
        boronic_simple = Chem.MolFromSmarts("[#6]-B(-[OH,O])-[OH,O]")
        
        patterns = [boronic_acid, boronic_ester, boronic_simple]
        
        for reactant in reactants:
            if reactant and any(reactant.HasSubstructMatch(p) for p in patterns if p):
                return True
        return False
    
    def _has_aryl_halide(self, reactants) -> bool:
        """Check for aromatic halide patterns"""
        aryl_halide = Chem.MolFromSmarts("c-[Cl,Br,I]")  # Aromatic carbon bonded to halide
        
        if not aryl_halide:
            return False
            
        for reactant in reactants:
            if reactant and reactant.HasSubstructMatch(aryl_halide):
                return True
        return False
    
    def _forms_new_biaryl_bond(self, product, reactants) -> bool:
        """Check if a new C(sp2)-C(sp2) bond is formed between aromatic rings"""
        biaryl_pattern = Chem.MolFromSmarts("c-c")  # Aromatic C-C bond
        
        if not biaryl_pattern or not product:
            return False
            
        # Count aromatic C-C bonds in product
        product_matches = len(product.GetSubstructMatches(biaryl_pattern))
        
        # Count aromatic C-C bonds in all reactants combined
        reactant_matches = 0
        for reactant in reactants:
            if reactant:
                reactant_matches += len(reactant.GetSubstructMatches(biaryl_pattern))
        
        # New biaryl bond formed if product has more aromatic C-C bonds
        return product_matches > reactant_matches
