"""Generated evaluation code for: Late stage Sonogashira coupling for biaryl-alkyne"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSonogashiraCoupling(BaseScoring):
    """
    Evaluates whether a Sonogashira coupling reaction occurs in the late stages of synthesis.
    Detects the formation of aryl-alkyne bonds from aryl halides and terminal alkynes.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)  # Late stage default
        
        # SMARTS patterns for Sonogashira coupling detection
        self.aryl_halide_pattern = "[cH0,c:1][I,Br,Cl]"  # Aryl halide
        self.terminal_alkyne_pattern = "[CH1]#C"  # Terminal alkyne
        self.aryl_alkyne_product_pattern = "[c:1]C#C"  # Aryl-alkyne product
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Sonogashira coupling doesn't occur
        
        if self.condition_type == "bool":
            return 1  # Reaction found
        else:
            # Late-stage coupling is preferred (higher depth fraction is better)
            return max(0, min(10, 10 * x))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents a Sonogashira coupling."""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if None in reactants or None in products:
                return False
            
            # Check for aryl halide and terminal alkyne in reactants
            has_aryl_halide = False
            has_terminal_alkyne = False
            
            aryl_halide_pattern = Chem.MolFromSmarts(self.aryl_halide_pattern)
            terminal_alkyne_pattern = Chem.MolFromSmarts(self.terminal_alkyne_pattern)
            aryl_alkyne_pattern = Chem.MolFromSmarts(self.aryl_alkyne_product_pattern)
            
            for reactant in reactants:
                if reactant.HasSubstructMatch(aryl_halide_pattern):
                    has_aryl_halide = True
                if reactant.HasSubstructMatch(terminal_alkyne_pattern):
                    has_terminal_alkyne = True
            
            # Check for aryl-alkyne formation in products
            has_aryl_alkyne_product = False
            for product in products:
                if product.HasSubstructMatch(aryl_alkyne_pattern):
                    has_aryl_alkyne_product = True
                    break
            
            # Sonogashira coupling detected if we have the characteristic reactants and product
            return has_aryl_halide and has_terminal_alkyne and has_aryl_alkyne_product
            
        except Exception:
            return False
