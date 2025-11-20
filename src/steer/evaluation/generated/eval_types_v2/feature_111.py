"""Generated evaluation code for: Early pyrazole ring construction via multicomponent reaction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyPyrazoleKnorrFormation(BaseScoring):
    """
    Evaluates synthesis routes for early pyrazole ring construction via multicomponent reaction.
    
    Checks for the formation of pyrazole rings (c1ccnnc1) through Knorr-type cyclization
    reactions, with preference for early-stage construction.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.method = config["parameters"]["method"]
        self.target_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Pyrazole formation doesn't occur
        
        if self.timing == "early":
            # Early formation is preferred - higher score for lower depth
            return max(0, 10 * (1 - x))
        else:
            # Late formation preferred
            return max(0, 10 * x)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms a pyrazole ring via Knorr-type mechanism"""
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Check if pyrazole ring is formed (present in products but not in reactants)
            reactant_has_pyrazole = any(mol.HasSubstructMatch(self.target_pattern) for mol in reactants)
            product_has_pyrazole = any(mol.HasSubstructMatch(self.target_pattern) for mol in products)
            
            if not product_has_pyrazole or reactant_has_pyrazole:
                return False
            
            # Check for multicomponent nature (3+ reactants for Knorr-type)
            if len(reactants) < 3:
                return False
                
            # Additional check for Knorr-type pattern indicators
            return self._is_knorr_type_mechanism(reactants, products)
            
        except Exception:
            return False
    
    def _is_knorr_type_mechanism(self, reactants, products) -> bool:
        """Check for characteristic patterns of Knorr-type pyrazole formation"""
        
        # Look for typical Knorr reactant patterns:
        # - Beta-dicarbonyl compounds or equivalents
        # - Hydrazine derivatives
        # - Aldehyde/ketone components
        
        hydrazine_pattern = Chem.MolFromSmarts("[NX3H2][NX3H2]")  # Hydrazine
        hydrazine_deriv_pattern = Chem.MolFromSmarts("[NX3H][NX3]")  # Hydrazine derivative
        dicarbonyl_pattern = Chem.MolFromSmarts("[CX3](=O)[CX4][CX3](=O)")  # Beta-dicarbonyl
        carbonyl_pattern = Chem.MolFromSmarts("[CX3]=O")  # General carbonyl
        
        has_hydrazine = any(mol.HasSubstructMatch(hydrazine_pattern) or 
                           mol.HasSubstructMatch(hydrazine_deriv_pattern) for mol in reactants)
        has_dicarbonyl = any(mol.HasSubstructMatch(dicarbonyl_pattern) for mol in reactants)
        carbonyl_count = sum(1 for mol in reactants if mol.HasSubstructMatch(carbonyl_pattern))
        
        # Knorr-type typically involves hydrazine + dicarbonyl or multiple carbonyl components
        return has_hydrazine and (has_dicarbonyl or carbonyl_count >= 2)
