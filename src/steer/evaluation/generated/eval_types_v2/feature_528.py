"""Generated evaluation code for: Early phosphonate formation via Michaelis-Arbuzov"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyPhosphonateFormation(BaseScoring):
    """
    Evaluates whether phosphonate formation via Michaelis-Arbuzov reaction occurs early in the synthesis.
    
    The Michaelis-Arbuzov reaction involves the reaction of a trialkyl phosphite with an alkyl halide
    to form a phosphonate ester, characterized by the formation of a P-C bond and elimination of 
    an alkyl halide.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "early")  # "early" means lower depth is better
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10 scale).
        For early timing preference, lower depth (earlier) gets higher score.
        """
        if x < 0:
            return 0  # Reaction doesn't occur in route
        
        if self.timing_preference == "early":
            # Early is better: depth 0 = score 10, depth 1 = score 0
            return 10 * (1 - x)
        else:
            # Late is better: depth 0 = score 0, depth 1 = score 10  
            return 10 * x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents a Michaelis-Arbuzov reaction.
        
        Michaelis-Arbuzov reaction pattern:
        - Reactants: trialkyl phosphite P(OR)3 + alkyl halide R'X
        - Product: phosphonate ester RP(=O)(OR)2 + alkyl halide RX
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if None in reactants or None in products:
                return False
            
            # Check for phosphite reactant P(OR)3
            phosphite_pattern = Chem.MolFromSmarts("[P]([O][C])([O][C])[O][C]")
            has_phosphite = any(mol.HasSubstructMatch(phosphite_pattern) for mol in reactants)
            
            # Check for alkyl halide reactant
            alkyl_halide_pattern = Chem.MolFromSmarts("[C][Cl,Br,I]")
            has_alkyl_halide = any(mol.HasSubstructMatch(alkyl_halide_pattern) for mol in reactants)
            
            # Check for phosphonate product P(=O)(OR)2C
            phosphonate_pattern = Chem.MolFromSmarts("[P](=[O])([O][C])([O][C])[C]")
            has_phosphonate = any(mol.HasSubstructMatch(phosphonate_pattern) for mol in products)
            
            # Michaelis-Arbuzov reaction should have all three components
            return has_phosphite and has_alkyl_halide and has_phosphonate
            
        except Exception:
            return False
