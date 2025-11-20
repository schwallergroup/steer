"""Generated evaluation code for: Late stage double cyclization cascade"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCyclizationCascade(BaseScoring):
    """
    Evaluates routes for late-stage double cyclization cascade reactions.
    Checks if the final step forms exactly 2 rings simultaneously in a cascade mechanism.
    """
    
    def __init__(self, config: Dict):
        self.target_ring_count = config["parameters"]["ring_count"]
        self.stage = config["parameters"]["stage"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No cascade cyclization found
        else:
            # Reward later stage cyclization (closer to 1.0 is better)
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction performs a double cyclization cascade"""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            product = Chem.MolFromSmiles(rxn[0])
            
            if not all([product] + reactants):
                return False
                
            # Count rings in reactants vs product
            reactant_rings = sum(mol.GetRingInfo().NumRings() for mol in reactants)
            product_rings = product.GetRingInfo().NumRings()
            
            # Check if exactly 2 rings are formed
            rings_formed = product_rings - reactant_rings
            if rings_formed != self.target_ring_count:
                return False
                
            # Check for cascade mechanism indicators
            return self._is_cascade_mechanism(reactants, product)
            
        except Exception:
            return False
    
    def _is_cascade_mechanism(self, reactants, product) -> bool:
        """
        Detect cascade mechanism by looking for:
        1. Multiple reactive sites in starting material
        2. Formation of fused/bridged ring systems
        3. Intramolecular cyclization patterns
        """
        # Look for typical cascade precursors with multiple reactive sites
        cascade_patterns = [
            # Diyne patterns for cascade cyclization
            "C#CC#C",
            # Diene patterns
            "C=CC=C",
            # Nucleophile-electrophile pairs for double cyclization
            "N[CH2][CH2]C=O",
            "NC(=O)C=C",
            # Azide-alkyne patterns for triazole formation cascades  
            "N=[N+]=[N-]",
            "C#C"
        ]
        
        # Check if any reactant contains cascade-prone patterns
        has_cascade_pattern = False
        for reactant in reactants:
            for pattern in cascade_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_cascade_pattern = True
                    break
        
        if not has_cascade_pattern:
            return False
            
        # Check for fused ring formation (indicator of cascade)
        ring_info = product.GetRingInfo()
        if ring_info.NumRings() < 2:
            return False
            
        # Look for shared atoms between rings (fused system)
        rings = ring_info.AtomRings()
        if len(rings) >= 2:
            for i, ring1 in enumerate(rings):
                for ring2 in rings[i+1:]:
                    if set(ring1) & set(ring2):  # Shared atoms = fused rings
                        return True
                        
        return False
