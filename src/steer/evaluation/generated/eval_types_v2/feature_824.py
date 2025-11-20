"""Generated evaluation code for: Late stage Suzuki coupling for biaryl formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzuki(BaseScoring):
    """
    Evaluates synthesis routes for late-stage Suzuki coupling reactions that form biaryl bonds.
    Rewards routes where the key biaryl bond is formed late in the synthesis using Suzuki chemistry.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)  # Default to late stage
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No Suzuki coupling found
        else:
            # Late-stage coupling is better, so higher depth values get higher scores
            if self.condition_type == "bool":
                return 1 if x >= self.target_depth else 0
            else:
                # Score increases with depth, penalize if too early
                if x >= self.target_depth:
                    return 1.0
                else:
                    return max(0, x / self.target_depth)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Suzuki coupling forming a biaryl bond"""
        metadata = d.get("metadata", {})
        
        # Check if reaction is classified as Suzuki coupling
        if metadata.get("policy_name") == "suzuki" or "suzuki" in metadata.get("reaction_type", "").lower():
            return True
            
        # Fallback: analyze the reaction SMILES for Suzuki pattern
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, product_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Check for biaryl formation
            biaryl_formed = self._has_biaryl_formation(reactants, product)
            
            # Check for Suzuki coupling indicators
            suzuki_indicators = self._has_suzuki_indicators(reactants, product)
            
            return biaryl_formed and suzuki_indicators
            
        except Exception:
            return False
    
    def _has_biaryl_formation(self, reactants, product):
        """Check if a biaryl bond is formed in this reaction"""
        # Biaryl pattern: two aromatic rings connected by single bond
        biaryl_patterns = [
            "[cR1]:[cR1]-[cR1]:[cR1]",  # Simple biaryl
            "c1ccccc1-c1ccccc1",       # Biphenyl-like
            "[c;R1]-[c;R1]",           # General aromatic-aromatic bond
        ]
        
        # Check if product has biaryl but reactants don't have this specific connection
        product_has_biaryl = any(product.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                               for pattern in biaryl_patterns)
        
        if not product_has_biaryl:
            return False
            
        # Verify that this biaryl bond was actually formed (not present in reactants)
        for reactant in reactants:
            if any(reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                   for pattern in biaryl_patterns):
                # If reactant already has biaryl, this might not be the formation step
                continue
        
        return True
    
    def _has_suzuki_indicators(self, reactants, product):
        """Check for chemical indicators of Suzuki coupling"""
        # Look for typical Suzuki coupling patterns
        boronic_acid_patterns = [
            "[C]-B(-O)(-O)",     # Boronic acid/ester
            "[C]-B(O)(O)",       # Boronic acid
            "[#6]-[#5]",         # C-B bond (general)
        ]
        
        halide_patterns = [
            "[C]-[Br]",          # Aryl bromide
            "[C]-[I]",           # Aryl iodide  
            "[C]-[Cl]",          # Aryl chloride
        ]
        
        # Check reactants for Suzuki coupling partners
        has_boron_reagent = any(
            any(reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                for pattern in boronic_acid_patterns)
            for reactant in reactants
        )
        
        has_halide_reagent = any(
            any(reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                for pattern in halide_patterns)
            for reactant in reactants
        )
        
        # Should have both coupling partners and product shouldn't have them
        product_lacks_boron = not any(product.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                                    for pattern in boronic_acid_patterns)
        product_lacks_halide = not any(product.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                                     for pattern in halide_patterns)
        
        return has_boron_reagent and has_halide_reagent and product_lacks_boron and product_lacks_halide
