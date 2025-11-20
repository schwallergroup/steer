"""Generated evaluation code for: Late stage alkyne introduction via Sonogashira"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAlkyneIntroduction(BaseScoring):
    """
    Evaluates routes for late-stage alkyne introduction via Sonogashira coupling.
    Checks if a Sonogashira reaction occurs at or after a specified depth.
    """
    
    def __init__(self, config: Dict):
        self.target_step = config["parameters"]["step"]
        self.timing = config["parameters"]["timing"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Sonogashira reaction doesn't occur
        
        # For late-stage timing, reward reactions at or after target step
        if self.timing == "late":
            if x >= self.target_step:
                return 1.0  # Perfect score for late-stage introduction
            else:
                return max(0, x / self.target_step)  # Partial score for earlier introduction
        else:
            # For other timing preferences, penalize deviation from target
            return max(0, 1.0 - abs(x - self.target_step) * 0.2)
    
    def hit_condition(self, d) -> bool:
        """
        Detects Sonogashira coupling by looking for:
        1. Formation of alkyne C-C bond
        2. Presence of halide leaving group in reactants
        3. Terminal alkyne in reactants
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Check for alkyne formation in product
            alkyne_pattern = Chem.MolFromSmarts("C#C")
            if not product.HasSubstructMatch(alkyne_pattern):
                return False
                
            # Check for terminal alkyne in reactants
            terminal_alkyne_pattern = Chem.MolFromSmarts("C#C[H]")
            has_terminal_alkyne = any(r.HasSubstructMatch(terminal_alkyne_pattern) for r in reactants)
            
            # Check for halide (typical Sonogashira substrate)
            halide_patterns = [
                Chem.MolFromSmarts("c-Br"),  # Aryl bromide
                Chem.MolFromSmarts("c-I"),   # Aryl iodide
                Chem.MolFromSmarts("c-Cl"),  # Aryl chloride
                Chem.MolFromSmarts("C-Br"),  # Alkyl bromide
                Chem.MolFromSmarts("C-I")    # Alkyl iodide
            ]
            
            has_halide = any(
                any(r.HasSubstructMatch(pattern) for r in reactants)
                for pattern in halide_patterns
            )
            
            # Sonogashira typically involves terminal alkyne + halide -> internal alkyne
            return has_terminal_alkyne and has_halide
            
        except Exception:
            return False
