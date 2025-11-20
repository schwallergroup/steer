"""Generated evaluation code for: Early quinoline ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class QuinolineRingFormation(BaseScoring):
    """
    Evaluates early quinoline ring formation in synthesis routes.
    
    Detects when a quinoline ring system is formed and scores based on timing.
    Early formation (closer to target) gets higher scores.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "early" or "late"
        self.direction = config["parameters"]["direction"]  # "formation" or "break"
        
        # Convert SMARTS to pattern for substructure matching
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "early":
            # Early formation preferred - higher score for smaller depth fractions
            return 1 - x
        else:
            # Late formation preferred - higher score for larger depth fractions
            return x
    
    def hit_condition(self, d) -> bool:
        """
        Check if quinoline ring formation occurs at this reaction step.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            # Split reaction SMILES
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product = rxn_parts[0]
            reactants = rxn_parts[1]
            
            # Parse molecules
            prod_mol = Chem.MolFromSmiles(product)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            
            if not prod_mol or not all(reactant_mols):
                return False
            
            # Check for quinoline ring formation
            if self.direction == "formation":
                return self._detect_ring_formation(prod_mol, reactant_mols)
            else:
                return self._detect_ring_break(prod_mol, reactant_mols)
                
        except Exception:
            return False
    
    def _detect_ring_formation(self, product, reactants):
        """
        Detect if quinoline ring is formed in this step.
        Ring should be present in product but not in any single reactant.
        """
        # Check if product contains quinoline pattern
        if not product.HasSubstructMatch(self.ring_pattern):
            return False
        
        # Check that no single reactant contains the complete quinoline pattern
        for reactant in reactants:
            if reactant.HasSubstructMatch(self.ring_pattern):
                return False
        
        return True
    
    def _detect_ring_break(self, product, reactants):
        """
        Detect if quinoline ring is broken in this step.
        Ring should be present in reactants but not in product.
        """
        # Check if any reactant contains quinoline pattern
        reactant_has_pattern = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants)
        
        if not reactant_has_pattern:
            return False
        
        # Check that product doesn't contain the pattern
        return not product.HasSubstructMatch(self.ring_pattern)
