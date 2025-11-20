"""Generated evaluation code for: Late stage amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Evaluates whether amide coupling reactions occur at a late stage in the synthesis.
    
    Checks for amide bond formation reactions and scores based on how late in the 
    synthesis route they occur, with preference for reactions happening after the
    specified stage threshold.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config["parameters"].get("stage_threshold", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No amide coupling found
        
        if x >= self.stage_threshold:
            return 10  # Perfect score for late-stage amide coupling
        else:
            # Linear scoring based on how close to late stage
            return 10 * (x / self.stage_threshold)
    
    def hit_condition(self, d) -> bool:
        """
        Check if the reaction is an amide coupling by detecting amide bond formation.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
        
        try:
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Count amide bonds in product and reactants
            amide_pattern = Chem.MolFromSmarts("[C,c](=[O,o])[N,n]")
            
            product_amides = len(product.GetSubstructMatches(amide_pattern))
            reactant_amides = sum(len(r.GetSubstructMatches(amide_pattern)) for r in reactants)
            
            # Amide coupling should increase the number of amide bonds
            return product_amides > reactant_amides
            
        except Exception:
            return False
