"""Generated evaluation code for: Early pyrazole core formation via Knorr synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyPyrazoleKnorrSynthesis(BaseScoring):
    """
    Evaluates synthesis routes for early pyrazole core formation via Knorr synthesis.
    Rewards routes where pyrazole rings are formed early in the synthetic sequence
    through Knorr condensation reactions.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.reaction_name = config["parameters"]["reaction_name"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        For early timing, lower depth fractions (earlier reactions) get higher scores.
        """
        if x < 0:
            return 0  # Condition not met
        
        if self.timing == "early":
            # Early formation preferred: score decreases with depth
            return max(0, 10 * (1 - x))
        else:
            # Late formation preferred: score increases with depth
            return min(10, 10 * x)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents pyrazole formation via Knorr synthesis.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Check if pyrazole ring is formed (present in products but not reactants)
            pyrazole_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
            pyrazole_in_products = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
            
            if not pyrazole_in_products or pyrazole_in_reactants:
                return False
            
            # Check for Knorr synthesis pattern
            return self._is_knorr_synthesis(reactants, products)
            
        except Exception:
            return False
    
    def _is_knorr_synthesis(self, reactants, products) -> bool:
        """
        Check if the reaction matches Knorr pyrazole synthesis pattern.
        Knorr synthesis typically involves condensation of β-dicarbonyl compounds
        with hydrazines or hydrazones.
        """
        # SMARTS patterns for typical Knorr synthesis components
        beta_dicarbonyl_pattern = Chem.MolFromSmarts("[C,c](=[O,o])[C,c][C,c](=[O,o])")  # β-dicarbonyl
        hydrazine_pattern = Chem.MolFromSmarts("[N,n][N,n]")  # hydrazine/hydrazone
        carbonyl_pattern = Chem.MolFromSmarts("[C,c]=[O,o]")  # general carbonyl
        
        if not beta_dicarbonyl_pattern or not hydrazine_pattern or not carbonyl_pattern:
            return False
        
        # Check if reactants contain typical Knorr synthesis components
        has_beta_dicarbonyl = any(mol.HasSubstructMatch(beta_dicarbonyl_pattern) for mol in reactants)
        has_hydrazine = any(mol.HasSubstructMatch(hydrazine_pattern) for mol in reactants)
        has_carbonyl = any(mol.HasSubstructMatch(carbonyl_pattern) for mol in reactants)
        
        # Knorr synthesis requires either:
        # 1. β-dicarbonyl + hydrazine, or
        # 2. hydrazine + two carbonyl-containing compounds
        knorr_pattern_1 = has_beta_dicarbonyl and has_hydrazine
        knorr_pattern_2 = has_hydrazine and len([mol for mol in reactants 
                                               if mol.HasSubstructMatch(carbonyl_pattern)]) >= 2
        
        return knorr_pattern_1 or knorr_pattern_2
