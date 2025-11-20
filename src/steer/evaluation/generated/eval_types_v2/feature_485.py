"""Generated evaluation code for: Late stage Sonogashira cross-coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSonogashira(BaseScoring):
    """
    Evaluates whether a Sonogashira cross-coupling reaction occurs at late stage.
    
    Sonogashira coupling involves C-C bond formation between a terminal alkyne 
    and an aryl/vinyl halide, typically forming C≡C-Ar bonds.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config["parameters"].get("stage_threshold", 0.3)
        # Sonogashira pattern: terminal alkyne + aryl/vinyl halide -> alkyne-aryl product
        self.terminal_alkyne_pattern = Chem.MolFromSmarts("C#[CH]")
        self.aryl_halide_pattern = Chem.MolFromSmarts("c[Cl,Br,I]")
        self.vinyl_halide_pattern = Chem.MolFromSmarts("C=C[Cl,Br,I]")
        self.alkyne_aryl_product = Chem.MolFromSmarts("C#C-c")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Sonogashira coupling doesn't occur
        
        # Late stage coupling is better (lower depth fraction)
        if x <= self.stage_threshold:
            return 10  # Excellent - very late stage
        elif x <= 0.5:
            return 8 - 4 * (x - self.stage_threshold) / (0.5 - self.stage_threshold)
        elif x <= 0.7:
            return 6 - 2 * (x - 0.5) / 0.2  
        else:
            return max(0, 4 - 4 * (x - 0.7) / 0.3)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Sonogashira coupling."""
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains alkyne-aryl bond
            has_alkyne_aryl = product.HasSubstructMatch(self.alkyne_aryl_product)
            if not has_alkyne_aryl:
                return False
            
            # Check if reactants contain terminal alkyne and aryl/vinyl halide
            has_terminal_alkyne = False
            has_halide = False
            
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.terminal_alkyne_pattern):
                    has_terminal_alkyne = True
                if (reactant.HasSubstructMatch(self.aryl_halide_pattern) or 
                    reactant.HasSubstructMatch(self.vinyl_halide_pattern)):
                    has_halide = True
            
            return has_terminal_alkyne and has_halide
            
        except Exception:
            return False
