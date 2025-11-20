"""Generated evaluation code for: Early stage ester reduction to alcohol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EsterReductionDepth(BaseScoring):
    """
    Evaluates the depth at which ester reduction to alcohol occurs in the synthesis route.
    Rewards early-stage ester reduction reactions based on a configurable depth threshold.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 7)
        self.timing = config.get("timing", "early")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ester reduction doesn't happen
        
        if self.timing == "early":
            # Reward early stage reactions (lower depth values)
            if x <= self.depth_threshold / 10:  # x is depth fraction
                return 10  # Perfect score for very early
            else:
                return max(0, 10 - (x * 10 - self.depth_threshold))
        else:
            # For late-stage preference
            return min(10, x * 10)
    
    def hit_condition(self, d):
        """
        Detects ester reduction to alcohol reactions by checking for:
        1. Ester functional group in product
        2. Primary alcohol in reactant
        3. Reduction conditions (hydride reagents)
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            prod = Chem.MolFromSmiles(rxn[0])
            reacts = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            # Define ester pattern (C(=O)O-C)
            ester_pattern = Chem.MolFromSmarts("C(=O)OC")
            # Define primary alcohol pattern
            alcohol_pattern = Chem.MolFromSmarts("CO")
            # Define common hydride reducing agents
            hydride_patterns = [
                Chem.MolFromSmarts("[Li+].[Al+3].[H-]"),  # LiAlH4
                Chem.MolFromSmarts("[Na+].[B+3].[H-]"),   # NaBH4
                Chem.MolFromSmarts("B([H-])([H-])([H-])[H-]"), # BH4-
                Chem.MolFromSmarts("[H-]")  # General hydride
            ]
            
            # Check if product has ester group
            has_ester_in_product = prod.HasSubstructMatch(ester_pattern)
            
            # Check if any reactant has primary alcohol
            has_alcohol_in_reactants = any(react.HasSubstructMatch(alcohol_pattern) for react in reacts)
            
            # Check for reducing agents in reactants
            has_reducing_agent = any(
                any(react.HasSubstructMatch(pattern) for pattern in hydride_patterns)
                for react in reacts
            )
            
            # Alternative check: look for ester to alcohol transformation by atom mapping
            if has_ester_in_product and has_alcohol_in_reactants:
                return self._verify_ester_reduction_mapping(prod, reacts)
            
            return has_ester_in_product and has_alcohol_in_reactants and has_reducing_agent
            
        except Exception:
            return False
    
    def _verify_ester_reduction_mapping(self, prod, reacts):
        """
        Verify ester reduction by checking atom mapping between ester carbon in product
        and alcohol carbon in reactants.
        """
        try:
            # Find ester carbons in product
            ester_pattern = Chem.MolFromSmarts("C(=O)OC")
            ester_matches = prod.GetSubstructMatches(ester_pattern)
            
            for match in ester_matches:
                ester_carbon_idx = match[0]  # First carbon in C(=O)OC
                ester_carbon = prod.GetAtomWithIdx(ester_carbon_idx)
                ester_map_num = ester_carbon.GetAtomMapNum()
                
                if ester_map_num > 0:
                    # Look for corresponding mapped atom in reactants
                    for react in reacts:
                        for atom in react.GetAtoms():
                            if atom.GetAtomMapNum() == ester_map_num:
                                # Check if this atom is part of alcohol group
                                alcohol_pattern = Chem.MolFromSmarts("CO")
                                if react.HasSubstructMatch(alcohol_pattern):
                                    alcohol_matches = react.GetSubstructMatches(alcohol_pattern)
                                    for alcohol_match in alcohol_matches:
                                        if atom.GetIdx() == alcohol_match[0]:
                                            return True
            return False
        except Exception:
            return False
