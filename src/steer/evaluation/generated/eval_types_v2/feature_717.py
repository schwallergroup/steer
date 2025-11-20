"""Generated evaluation code for: Late stage Suzuki coupling for pyridine installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiPyridine(BaseScoring):
    """
    Evaluates routes for late-stage Suzuki coupling reactions that install pyridine or thiophene rings.
    Rewards routes where these heterocyclic couplings occur later in the synthesis (closer to the target).
    """
    
    def __init__(self, config: Dict):
        self.coupling_partners = config.get("coupling_partners", ["pyridine", "thiophene"])
        # SMARTS patterns for pyridine and thiophene rings
        self.pyridine_pattern = "c1ccncc1"
        self.thiophene_pattern = "c1ccsc1"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling with target heterocycles doesn't happen
        else:
            return 1 - x  # Later stage coupling is better (lower depth fraction = higher score)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Suzuki coupling installing pyridine or thiophene"""
        metadata = d.get("metadata", {})
        
        # Check if this is identified as a Suzuki coupling
        if not self._is_suzuki_coupling(metadata):
            return False
            
        # Check if pyridine/thiophene is being installed
        return self._installs_target_heterocycle(metadata)
    
    def _is_suzuki_coupling(self, metadata) -> bool:
        """Identify Suzuki coupling reactions"""
        reaction_smiles = metadata.get("mapped_reaction_smiles", "")
        if not reaction_smiles:
            return False
            
        # Look for Suzuki coupling indicators in reaction SMILES
        # Typically involves B(OH)2, B(OR)2, or BF3K groups and halides/triflates
        boronic_patterns = ["B(O)(O)", "B(OC)(OC)", "BF3", "B(O)O"]
        halide_patterns = ["Br", "I", "Cl"]
        
        rxn_parts = reaction_smiles.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        
        # Check for boronic acid/ester and halide presence
        has_boron = any(pattern in reactants for pattern in boronic_patterns)
        has_halide = any(pattern in reactants for pattern in halide_patterns)
        
        return has_boron and has_halide
    
    def _installs_target_heterocycle(self, metadata) -> bool:
        """Check if the reaction installs pyridine or thiophene rings"""
        reaction_smiles = metadata.get("mapped_reaction_smiles", "")
        if not reaction_smiles:
            return False
            
        try:
            rxn_parts = reaction_smiles.split(">>")
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Create patterns for target heterocycles
            patterns = []
            if "pyridine" in self.coupling_partners:
                patterns.append(Chem.MolFromSmarts(self.pyridine_pattern))
            if "thiophene" in self.coupling_partners:
                patterns.append(Chem.MolFromSmarts(self.thiophene_pattern))
            
            # Check if product contains the heterocycle
            product_has_heterocycle = any(product.HasSubstructMatch(pattern) for pattern in patterns if pattern)
            
            # Check if any single reactant contains the heterocycle (coupling partner)
            reactant_has_heterocycle = any(
                any(reactant.HasSubstructMatch(pattern) for pattern in patterns if pattern)
                for reactant in reactants
            )
            
            # True if product has heterocycle and it came from a reactant (coupling installation)
            return product_has_heterocycle and reactant_has_heterocycle
            
        except Exception:
            return False
