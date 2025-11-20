"""Generated evaluation code for: Late stage sulfamoylation of primary alcohol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSulfamoylation(BaseScoring):
    """
    Evaluates whether late-stage sulfamoylation of primary alcohol occurs in the synthesis route.
    Checks for conversion of primary alcohol to sulfamate ester using sulfamoyl chloride or equivalent.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10), favoring late-stage reactions"""
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            return 1 - x  # Later stage is better (closer to 1.0)
    
    def hit_condition(self, d) -> bool:
        """Check if a reaction node represents sulfamoylation of primary alcohol"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if sulfamoyl chloride or equivalent is present in reactants
            sulfamoyl_patterns = [
                "ClS(=O)(=O)N",  # sulfamoyl chloride
                "ClS(=O)(=O)NC",  # N-substituted sulfamoyl chloride
                "ClS(=O)(=O)N(C)C",  # N,N-disubstituted sulfamoyl chloride
            ]
            
            has_sulfamoyl_reagent = False
            for reactant in reactants:
                for pattern in sulfamoyl_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_sulfamoyl_reagent = True
                        break
                if has_sulfamoyl_reagent:
                    break
            
            if not has_sulfamoyl_reagent:
                return False
            
            # Check for primary alcohol in reactants and sulfamate in product
            primary_alcohol_pattern = "[CH2]-O[H]"  # Primary alcohol
            sulfamate_pattern = "CO[S](=O)(=O)N"  # Sulfamate ester
            
            has_primary_alcohol = False
            for reactant in reactants:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(primary_alcohol_pattern)):
                    has_primary_alcohol = True
                    break
            
            has_sulfamate = product.HasSubstructMatch(Chem.MolFromSmarts(sulfamate_pattern))
            
            return has_sulfamoyl_reagent and has_primary_alcohol and has_sulfamate
            
        except Exception:
            return False
