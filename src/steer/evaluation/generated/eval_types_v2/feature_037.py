"""Generated evaluation code for: Late stage ester formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageEsterFormation(BaseScoring):
    """
    Evaluates whether ester formation occurs at a late stage in the synthesis route.
    Detects esterification reactions involving carboxylic acids and scores based on 
    how late in the route the reaction occurs.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)  # Late stage default
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Esterification doesn't happen
        
        if self.condition_type == "bool":
            # For late stage, we want depth > 0.7 (closer to 1.0 means later)
            return 1 if x > 0.7 else 0
        else:
            # Continuous scoring - reward later stages more
            if x > self.target_depth:
                return 1.0  # Perfect score for reactions at target depth or later
            else:
                return x / self.target_depth  # Scale score based on how close to target
    
    def hit_condition(self, d):
        """
        Detects esterification reactions involving carboxylic acids.
        Looks for formation of ester bonds from carboxylic acid substrates.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for carboxylic acid in reactants
            carboxylic_acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")  # COOH pattern
            has_carboxylic_acid = any(mol.HasSubstructMatch(carboxylic_acid_pattern) for mol in reactants)
            
            if not has_carboxylic_acid:
                return False
            
            # Check for ester formation in products
            ester_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][CX4]")  # COO-C pattern (ester)
            has_ester = any(mol.HasSubstructMatch(ester_pattern) for mol in products)
            
            if not has_ester:
                return False
            
            # Additional check: ensure we're not just hydrolyzing an ester (reverse reaction)
            # Count esters in reactants vs products
            reactant_ester_count = sum(len(mol.GetSubstructMatches(ester_pattern)) for mol in reactants)
            product_ester_count = sum(len(mol.GetSubstructMatches(ester_pattern)) for mol in products)
            
            # Ester formation should increase ester count
            return product_ester_count > reactant_ester_count
            
        except Exception:
            return False
