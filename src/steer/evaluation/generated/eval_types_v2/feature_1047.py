"""Generated evaluation code for: Early bicyclic core assembly via cycloaddition"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyBicyclicCycloaddition(BaseScoring):
    """
    Evaluates whether bicyclic core assembly via cycloaddition occurs early in the synthesis route.
    Checks for cycloaddition reactions that form bicyclic systems within the first half of the route.
    """
    
    def __init__(self, config: Dict):
        self.target_step = config["parameters"]["ring_formation_step"]
        self.total_steps = config["parameters"]["total_steps"]
        self.formation_type = config["parameters"]["formation_type"]
        
        # Define bicyclic patterns to detect
        self.bicyclic_patterns = [
            "[R]1~[R]~[R]~[R]2~[R]~[R]~[R]~[R]~1~2",  # Generic bicyclic
            "[R]1~[R]~[R]~[R]2~[R]~[R]~[R]1~2",        # 6-5 fused
            "[R]1~[R]~[R]~[R]~[R]2~[R]~[R]~[R]1~2",    # 6-6 fused
            "C1CCC2CCCCC12",  # Decalin-like
            "C1CCC2CCCC12",   # Bicyclic saturated
        ]
        
    def route_scoring(self, x) -> float:
        """
        Score based on how early the bicyclic cycloaddition occurs.
        Earlier formation (lower depth) gives higher score.
        """
        if x < 0:
            return 0  # No bicyclic cycloaddition found
        
        # Convert depth fraction to step number
        actual_step = x * self.total_steps
        
        # Score higher for earlier formation
        if actual_step <= self.target_step:
            return 10  # Perfect score for early formation
        else:
            # Penalize late formation
            delay = actual_step - self.target_step
            penalty = min(delay / self.total_steps * 10, 10)
            return max(0, 10 - penalty)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction is a cycloaddition that forms a bicyclic system.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check if this is a cycloaddition (typically 2 reactants -> 1 product with ring formation)
            if not self._is_cycloaddition_pattern(reactants, products):
                return False
            
            # Check if bicyclic system is formed
            return self._forms_bicyclic_system(reactants, products)
            
        except Exception:
            return False
    
    def _is_cycloaddition_pattern(self, reactants, products) -> bool:
        """Check if reaction pattern matches cycloaddition characteristics."""
        # Typically 2 reactants combining to form 1 main product
        if len(reactants) != 2:
            return False
        
        # Count rings before and after
        reactant_rings = sum(mol.GetRingInfo().NumRings() for mol in reactants)
        product_rings = sum(mol.GetRingInfo().NumRings() for mol in products)
        
        # Ring formation should occur (at least one new ring)
        return product_rings > reactant_rings
    
    def _forms_bicyclic_system(self, reactants, products) -> bool:
        """Check if the reaction forms a bicyclic system."""
        # Check if any reactant already has bicyclic system
        reactant_has_bicyclic = any(self._has_bicyclic_system(mol) for mol in reactants)
        
        # Check if any product has bicyclic system
        product_has_bicyclic = any(self._has_bicyclic_system(mol) for mol in products)
        
        # Bicyclic system should be formed (not just preserved)
        return product_has_bicyclic and not reactant_has_bicyclic
    
    def _has_bicyclic_system(self, mol) -> bool:
        """Check if molecule contains a bicyclic system."""
        if not mol or mol.GetRingInfo().NumRings() < 2:
            return False
        
        # Check against bicyclic patterns
        for pattern in self.bicyclic_patterns:
            try:
                patt_mol = Chem.MolFromSmarts(pattern)
                if patt_mol and mol.HasSubstructMatch(patt_mol):
                    return True
            except:
                continue
        
        # Check for fused ring systems
        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() >= 2:
            rings = ring_info.AtomRings()
            # Check if any two rings share atoms (fused system)
            for i, ring1 in enumerate(rings):
                for ring2 in rings[i+1:]:
                    if set(ring1) & set(ring2):  # Shared atoms = fused rings
                        return True
        
        return False
