"""Generated evaluation code for: Late stage sulfamate ester formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSulfamateFormation(BaseScoring):
    """
    Evaluates routes for late-stage sulfamate ester formation.
    
    Detects sulfamate ester formation reactions and scores based on how late 
    in the synthesis the sulfamate functional group is installed. Higher scores
    are given for later installation (closer to final product).
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)  # Default to late stage
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Sulfamate formation doesn't happen
        else:
            # Late-stage formation is better (higher depth fraction = better score)
            # Scale to 0-10 range where 10 is best (latest stage)
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves sulfamate ester formation.
        
        Looks for:
        1. Formation of S(=O)2-N bond from sulfamoyl chloride or similar
        2. Product contains sulfamate ester pattern
        3. Reactants contain alcohol and sulfamoyl chloride/derivative
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[1]
            reactants = rxn_parts[0].split(".")
            
            # Parse molecules
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r) for r in reactants if r.strip()]
            
            if not prod_mol or not all(react_mols):
                return False
            
            # Check for sulfamate ester formation
            return self._is_sulfamate_formation(react_mols, prod_mol)
            
        except Exception:
            return False
    
    def _is_sulfamate_formation(self, reactants, product):
        """
        Detect sulfamate ester formation by checking:
        1. Product has sulfamate ester pattern
        2. Reactants have alcohol and sulfamoyl compound
        """
        # Sulfamate ester patterns
        sulfamate_patterns = [
            "COS(=O)(=O)N",  # Basic sulfamate ester
            "[CH2]OS(=O)(=O)N",  # Primary alcohol sulfamate
            "[CH]OS(=O)(=O)N",   # Secondary alcohol sulfamate
        ]
        
        # Check if product contains sulfamate ester
        has_sulfamate_product = False
        for pattern in sulfamate_patterns:
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol and product.HasSubstructMatch(pattern_mol):
                has_sulfamate_product = True
                break
        
        if not has_sulfamate_product:
            return False
        
        # Check reactants for alcohol and sulfamoyl compound
        has_alcohol = False
        has_sulfamoyl = False
        
        alcohol_pattern = Chem.MolFromSmarts("[CH2,CH,CH3]O")  # Alcohol group
        sulfamoyl_patterns = [
            "ClS(=O)(=O)N",  # Sulfamoyl chloride
            "S(=O)(=O)N",    # General sulfamoyl group
        ]
        
        for reactant in reactants:
            # Check for alcohol
            if alcohol_pattern and reactant.HasSubstructMatch(alcohol_pattern):
                has_alcohol = True
            
            # Check for sulfamoyl compound
            for sulf_pattern in sulfamoyl_patterns:
                sulf_mol = Chem.MolFromSmarts(sulf_pattern)
                if sulf_mol and reactant.HasSubstructMatch(sulf_mol):
                    has_sulfamoyl = True
                    break
        
        return has_alcohol and has_sulfamoyl
