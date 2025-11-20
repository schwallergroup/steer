"""Generated evaluation code for: Late stage amide coupling convergent assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Evaluates synthesis routes for late-stage convergent amide coupling reactions.
    Checks if an amide bond formation occurs at a specified depth with the required
    number of fragments being assembled.
    """
    
    def __init__(self, config: Dict):
        self.target_stage = config["parameters"]["stage"]  # "late" 
        self.fragment_count = config["parameters"]["fragment_count"]  # 2
        self.coupling_type = config["parameters"]["coupling_reaction_type"]  # "amide_formation"
        
        # Define amide formation patterns - look for amide bond in product
        self.amide_pattern = Chem.MolFromSmarts("[C,c](=O)[NH,N]")
        
        # Common amide coupling reagent patterns
        self.coupling_reagents = [
            "[C,c](=O)O",  # Carboxylic acid
            "[C,c](=O)Cl",  # Acyl chloride
            "[NH2,NH]",  # Amine
            "C1=CC=NC=C1.C1=CC=CC=N1",  # PyBOP-like reagents (simplified)
        ]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        
        if self.target_stage == "late":
            # For late stage, prefer early depth (closer to 0)
            # Convert to 0-10 scale where lower depth = higher score
            return max(0, 10 * (1 - x))
        else:
            # For early/mid stage, could prefer different depths
            return 10 * (1 - abs(x - 0.5))  # Prefer middle depths
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents an amide coupling"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            product_smiles, reactants_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or len(reactants) < self.fragment_count:
                return False
                
            # Check if product contains amide bond
            if not product.HasSubstructMatch(self.amide_pattern):
                return False
                
            # Check if this is a coupling reaction by verifying:
            # 1. Product has amide bond that reactants don't have (newly formed)
            # 2. Reactants contain appropriate functional groups
            
            # Count amides in product vs reactants
            product_amides = len(product.GetSubstructMatches(self.amide_pattern))
            reactant_amides = sum(len(r.GetSubstructMatches(self.amide_pattern)) 
                                for r in reactants if r is not None)
            
            # New amide bond formed
            if product_amides <= reactant_amides:
                return False
                
            # Check for appropriate coupling partners
            has_acid_component = False
            has_amine_component = False
            
            acid_pattern = Chem.MolFromSmarts("[C,c](=O)[OH,Cl]")
            amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
            
            for reactant in reactants:
                if reactant is None:
                    continue
                if reactant.HasSubstructMatch(acid_pattern):
                    has_acid_component = True
                if reactant.HasSubstructMatch(amine_pattern):
                    has_amine_component = True
                    
            return has_acid_component and has_amine_component
            
        except Exception:
            return False
