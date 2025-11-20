"""Generated evaluation code for: Cyclopropanation via Corey-Chaykovsky reaction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CoreyChakovskyCyclopropanation(BaseScoring):
    """
    Detects Corey-Chaykovsky cyclopropanation reactions, specifically looking for
    sulfur ylide-mediated cyclopropane formation from alpha,beta-unsaturated esters.
    
    The reaction involves:
    1. Sulfur ylide (typically from dimethylsulfonium methylide)
    2. Alpha,beta-unsaturated ester substrate
    3. Formation of cyclopropane ring
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10)"""
        if x < 0:
            return 0  # Reaction not found
        
        if self.condition_type == "bool":
            return 10  # Reaction found
        else:
            # Earlier in synthesis is better (lower depth fraction = higher score)
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """Check if a single reaction node represents Corey-Chaykovsky cyclopropanation"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            product_smiles, reactants_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for cyclopropane formation in product
            cyclopropane_pattern = Chem.MolFromSmarts("[C;R3]1[C;R3][C;R3]1")
            if not product.HasSubstructMatch(cyclopropane_pattern):
                return False
            
            # Check for alpha,beta-unsaturated ester in reactants
            unsaturated_ester_pattern = Chem.MolFromSmarts("C=C-C(=O)O")
            has_unsaturated_ester = any(r.HasSubstructMatch(unsaturated_ester_pattern) for r in reactants)
            
            # Check for sulfur ylide or methylating agent (sulfur-containing reagent)
            sulfur_ylide_patterns = [
                Chem.MolFromSmarts("[S+]([CH3])([CH3])[CH2-]"),  # Dimethylsulfonium methylide
                Chem.MolFromSmarts("[S+]([CH3])([CH3])[CH2]"),   # Sulfonium salt
                Chem.MolFromSmarts("S([CH3])([CH3])=C"),         # Sulfur ylide resonance form
                Chem.MolFromSmarts("[S]"),                       # Any sulfur-containing reagent
            ]
            
            has_sulfur_reagent = any(
                r.HasSubstructMatch(pattern) 
                for r in reactants 
                for pattern in sulfur_ylide_patterns 
                if pattern is not None
            )
            
            # Additional check: ensure cyclopropane wasn't present in reactants
            cyclopropane_in_reactants = any(r.HasSubstructMatch(cyclopropane_pattern) for r in reactants)
            
            # Must have: unsaturated ester substrate, sulfur reagent, cyclopropane formation
            return (has_unsaturated_ester and 
                   has_sulfur_reagent and 
                   not cyclopropane_in_reactants)
                   
        except Exception:
            return False
