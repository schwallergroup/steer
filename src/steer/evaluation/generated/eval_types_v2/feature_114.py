"""Generated evaluation code for: Late stage lactam ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageLactamFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage lactam ring formation.
    Checks if a 6-membered lactam ring is formed in the later stages of synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_size = config["parameters"].get("ring_size", 6)
        self.timing = config["parameters"].get("timing", "late")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No lactam formation detected
        
        if self.timing == "late":
            # Reward later formation (higher depth fraction is better)
            return 10 * x
        else:
            # For early timing, lower depth fraction is better
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves lactam ring formation"""
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
                
            # Check for lactam ring formation
            return self._detects_lactam_formation(reactants, products)
            
        except Exception:
            return False
    
    def _detects_lactam_formation(self, reactants, products) -> bool:
        """Detect if lactam ring is formed in this reaction"""
        
        # Define lactam patterns for different ring sizes
        lactam_patterns = {
            5: "[#6]1[#6][#6][#6][#7]([#6]([#6])=[#8])1",  # 5-membered lactam
            6: "[#6]1[#6][#6][#6][#6][#7]([#6]=[#8])1",     # 6-membered lactam
            7: "[#6]1[#6][#6][#6][#6][#6][#7]([#6]=[#8])1"  # 7-membered lactam
        }
        
        pattern = lactam_patterns.get(self.ring_size)
        if not pattern:
            return False
            
        lactam_pattern = Chem.MolFromSmarts(pattern)
        if lactam_pattern is None:
            return False
        
        # Check if lactam is present in products but not in reactants
        product_has_lactam = any(mol.HasSubstructMatch(lactam_pattern) for mol in products)
        reactant_has_lactam = any(mol.HasSubstructMatch(lactam_pattern) for mol in reactants)
        
        # Also check for general amide formation that could indicate lactam cyclization
        if product_has_lactam and not reactant_has_lactam:
            return True
            
        # Additional check: look for intramolecular cyclization patterns
        # Check if we have an amide bond formation with ring closure
        amide_pattern = Chem.MolFromSmarts("[#7]-[#6]=[#8]")
        if amide_pattern is None:
            return False
            
        # Check for new amide formation
        product_amides = sum(len(mol.GetSubstructMatches(amide_pattern)) for mol in products)
        reactant_amides = sum(len(mol.GetSubstructMatches(amide_pattern)) for mol in reactants)
        
        # If we have new amide formation and new lactam, this is likely lactam formation
        return (product_amides > reactant_amides) and product_has_lactam and not reactant_has_lactam
