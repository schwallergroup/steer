"""Generated evaluation code for: Late stage Sonogashira coupling for alkyne installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStagesonogashira(BaseScoring):
    """
    Evaluates whether a Sonogashira coupling reaction occurs at late stage in the synthesis route.
    Checks for C-C bond formation between an alkyne and aryl halide (especially aryl iodide).
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "late")  # "late" prefers later stages
        self.substrate_pattern = config.get("substrate_pattern", "aryl_iodide")
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score.
        For late-stage preference: later reactions (higher x) get better scores.
        """
        if x < 0:
            return 0  # Reaction doesn't occur
        
        if self.timing_preference == "late":
            return 1 - x  # Later stages get higher scores (closer to 1)
        else:
            return x  # Earlier stages get higher scores
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents a Sonogashira coupling.
        Looks for alkyne + aryl halide -> aryl alkyne pattern.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_str, products_str = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_str.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_str.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for Sonogashira coupling patterns
            return self._is_sonogashira_coupling(reactants, products)
            
        except Exception:
            return False
    
    def _is_sonogashira_coupling(self, reactants, products) -> bool:
        """
        Detect Sonogashira coupling by checking for:
        1. Terminal alkyne in reactants
        2. Aryl halide in reactants  
        3. Aryl alkyne in products
        """
        # Terminal alkyne pattern: C#C[H] or variations
        terminal_alkyne_patterns = [
            Chem.MolFromSmarts("[C]#[CH]"),
            Chem.MolFromSmarts("[CH]#[C]"),
            Chem.MolFromSmarts("C#C")
        ]
        
        # Aryl halide patterns (prioritize iodide if specified)
        if self.substrate_pattern == "aryl_iodide":
            aryl_halide_patterns = [
                Chem.MolFromSmarts("c[I]"),  # Aryl iodide
                Chem.MolFromSmarts("c[Br]"), # Also accept bromide
                Chem.MolFromSmarts("c[Cl]")  # Also accept chloride
            ]
        else:
            aryl_halide_patterns = [
                Chem.MolFromSmarts("c[I,Br,Cl]")  # Any aryl halide
            ]
        
        # Aryl alkyne product pattern
        aryl_alkyne_patterns = [
            Chem.MolFromSmarts("c-C#C"),
            Chem.MolFromSmarts("c[C]#[C]")
        ]
        
        # Check reactants for alkyne and aryl halide
        has_alkyne = any(
            any(mol.HasSubstructMatch(pattern) for pattern in terminal_alkyne_patterns if pattern)
            for mol in reactants
        )
        
        has_aryl_halide = any(
            any(mol.HasSubstructMatch(pattern) for pattern in aryl_halide_patterns if pattern)
            for mol in reactants
        )
        
        # Check products for aryl alkyne
        has_aryl_alkyne = any(
            any(mol.HasSubstructMatch(pattern) for pattern in aryl_alkyne_patterns if pattern)
            for mol in products
        )
        
        return has_alkyne and has_aryl_halide and has_aryl_alkyne
