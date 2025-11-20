"""Generated evaluation code for: Late stage ester formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageEsterFormation(BaseScoring):
    """
    Evaluates whether esterification reactions occur at late stages (shallow depths)
    in the synthesis route. Rewards routes where ester formation happens near the
    final product (depth <= depth_threshold).
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 2)
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10 scale).
        Rewards late-stage esterification (low depth values).
        """
        if x < 0:
            return 0  # No esterification found
        
        # Convert depth fraction to actual depth and compare to threshold
        # x is depth_fraction, lower values mean later in synthesis
        if x <= (self.depth_threshold / 10.0):  # Assuming max depth ~10
            return 10  # Perfect score for very late stage
        else:
            # Penalize early esterification
            return max(0, 10 - (x * 50))  # Scale penalty
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents an esterification reaction.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if None in products or None in reactants:
                return False
            
            # Check for ester formation: look for ester bonds in products that aren't in reactants
            ester_pattern = Chem.MolFromSmarts("[C](=[O])[O][C]")
            
            # Count ester groups in products
            product_ester_count = sum(len(mol.GetSubstructMatches(ester_pattern)) 
                                    for mol in products if mol is not None)
            
            # Count ester groups in reactants
            reactant_ester_count = sum(len(mol.GetSubstructMatches(ester_pattern)) 
                                     for mol in reactants if mol is not None)
            
            # Esterification if more ester bonds in products than reactants
            return product_ester_count > reactant_ester_count
            
        except Exception:
            return False
