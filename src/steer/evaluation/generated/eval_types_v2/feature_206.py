"""Generated evaluation code for: Late heterocyclic core formation via cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AminopyrimidineFormation(BaseScoring):
    """
    Evaluates routes for late-stage aminopyrimidine core formation via base-mediated cyclization.
    Detects when an aminopyrimidine ring is formed and scores based on timing in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "late")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing_preference == "late":
            # Reward later formation (higher depth fraction is better)
            return 10 * x
        elif self.timing_preference == "early":
            # Reward earlier formation (lower depth fraction is better)
            return 10 * (1 - x)
        else:
            # Neutral - just check if it happens
            return 5
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms an aminopyrimidine ring via cyclization"""
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check if aminopyrimidine is formed
            aminopyrimidine_pattern = Chem.MolFromSmarts("[#7]-c1nc([#7])ncn1")
            alt_aminopyrimidine_pattern = Chem.MolFromSmarts("[#7]-c1ncnc([#7])n1")
            
            # Check if any product contains aminopyrimidine
            product_has_aminopyrimidine = any(
                mol.HasSubstructMatch(aminopyrimidine_pattern) or 
                mol.HasSubstructMatch(alt_aminopyrimidine_pattern)
                for mol in products
            )
            
            if not product_has_aminopyrimidine:
                return False
            
            # Check if reactants lack the complete aminopyrimidine (indicating formation)
            reactants_have_complete_aminopyrimidine = any(
                mol.HasSubstructMatch(aminopyrimidine_pattern) or 
                mol.HasSubstructMatch(alt_aminopyrimidine_pattern)
                for mol in reactants
            )
            
            if reactants_have_complete_aminopyrimidine:
                return False  # Ring already present, not formation
            
            # Additional check for cyclization pattern - look for ring count increase
            reactant_ring_counts = [mol.GetRingInfo().NumRings() for mol in reactants]
            product_ring_counts = [mol.GetRingInfo().NumRings() for mol in products]
            
            total_reactant_rings = sum(reactant_ring_counts)
            total_product_rings = sum(product_ring_counts)
            
            # Ring formation should increase ring count
            return total_product_rings > total_reactant_rings
            
        except Exception:
            return False
