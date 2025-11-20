"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSuzukiStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two fragments are coupled
    via Suzuki-Miyaura cross-coupling reaction at a specific depth.
    
    Checks for the presence of Suzuki coupling (C-C bond formation between
    aryl/vinyl boronic acid derivatives and aryl/vinyl halides) and validates
    that exactly two complex fragments are being joined.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["parameters"].get("fragment_count", 2)
        self.coupling_reaction = config["parameters"].get("coupling_reaction", "Suzuki-Miyaura")
        self.convergent = config["parameters"].get("convergent", True)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling not found
        else:
            # Earlier convergent coupling is better (closer to final step)
            # x is depth fraction, so smaller values = earlier in synthesis
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Suzuki-Miyaura coupling between two fragments"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            product_smiles, reactants_smiles = mapped_rxn.split(">>")
            reactant_list = reactants_smiles.split(".")
            
            # Must have exactly the specified number of fragments
            if len(reactant_list) != self.fragment_count:
                return False
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_list]
            
            if not product or any(r is None for r in reactants):
                return False
            
            # Check for Suzuki coupling pattern
            if not self._is_suzuki_coupling(product, reactants):
                return False
            
            # Verify convergent strategy - both fragments should be reasonably complex
            if self.convergent:
                return self._verify_convergent_fragments(reactants)
            
            return True
            
        except Exception:
            return False
    
    def _is_suzuki_coupling(self, product, reactants) -> bool:
        """Detect Suzuki-Miyaura coupling pattern"""
        # Look for boronic acid/ester patterns in reactants
        boron_patterns = [
            "[#6]-B(-O)-O",  # Boronic acid
            "[#6]-B1OC[CH2]O1",  # Boronic ester (pinacol)
            "[#6]-B(O)O"  # Simple boronic acid
        ]
        
        # Look for halide patterns (Br, I, sometimes Cl)
        halide_patterns = [
            "[#6]-Br",
            "[#6]-I", 
            "[#6]-Cl"
        ]
        
        has_boron = False
        has_halide = False
        
        for reactant in reactants:
            # Check for boron-containing reactant
            for boron_pattern in boron_patterns:
                boron_query = Chem.MolFromSmarts(boron_pattern)
                if boron_query and reactant.HasSubstructMatch(boron_query):
                    has_boron = True
                    break
            
            # Check for halide-containing reactant
            for halide_pattern in halide_patterns:
                halide_query = Chem.MolFromSmarts(halide_pattern)
                if halide_query and reactant.HasSubstructMatch(halide_query):
                    has_halide = True
                    break
        
        # Suzuki coupling requires both boron and halide components
        return has_boron and has_halide
    
    def _verify_convergent_fragments(self, reactants) -> bool:
        """Verify that fragments are reasonably complex for convergent strategy"""
        min_complexity_atoms = 6  # Minimum atoms for a "complex" fragment
        
        complex_fragments = 0
        for reactant in reactants:
            if reactant.GetNumAtoms() >= min_complexity_atoms:
                complex_fragments += 1
        
        # For true convergent synthesis, both fragments should be reasonably complex
        return complex_fragments >= 2
