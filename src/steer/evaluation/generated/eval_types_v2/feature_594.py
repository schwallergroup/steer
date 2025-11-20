"""Generated evaluation code for: Early Sonogashira coupling for C-C bond formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SonogashiraCoupling(BaseScoring):
    """
    Evaluates synthesis routes for early-stage Sonogashira coupling reactions.
    Rewards routes where Sonogashira coupling occurs early (within depth threshold).
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 5)
        self.stage = config.get("stage", "early")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Sonogashira coupling doesn't occur
        
        if self.stage == "early":
            # Reward early-stage coupling (lower depth is better)
            if x <= self.depth_threshold / 10.0:  # x is depth fraction
                return 10 * (1 - x)  # Higher score for earlier occurrence
            else:
                return 0  # Too late, no reward
        else:
            # For non-early stage, standard scoring
            return 5 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Detects Sonogashira coupling by identifying terminal alkyne + aryl halide pattern.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for Sonogashira coupling pattern
            return self._detect_sonogashira_pattern(product, reactants)
            
        except Exception:
            return False
    
    def _detect_sonogashira_pattern(self, product, reactants) -> bool:
        """
        Detects Sonogashira coupling by checking for:
        1. Terminal alkyne in reactants
        2. Aryl halide in reactants  
        3. Alkyne-aryl bond formation in product
        """
        # Terminal alkyne pattern
        terminal_alkyne = Chem.MolFromSmarts("[C]#[CH]")
        # Aryl halide patterns (Br, I, Cl on aromatic carbon)
        aryl_halide = Chem.MolFromSmarts("c[Br,I,Cl]")
        # Product alkyne-aryl pattern
        alkyne_aryl = Chem.MolFromSmarts("c[C]#[C]")
        
        if not all([terminal_alkyne, aryl_halide, alkyne_aryl]):
            return False
        
        # Check if product contains alkyne-aryl bond
        has_alkyne_aryl_product = product.HasSubstructMatch(alkyne_aryl)
        
        if not has_alkyne_aryl_product:
            return False
        
        # Check if reactants contain terminal alkyne and aryl halide
        has_terminal_alkyne = any(r.HasSubstructMatch(terminal_alkyne) for r in reactants)
        has_aryl_halide = any(r.HasSubstructMatch(aryl_halide) for r in reactants)
        
        return has_terminal_alkyne and has_aryl_halide
