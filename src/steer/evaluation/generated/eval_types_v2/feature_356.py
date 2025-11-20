"""Generated evaluation code for: Late stage Williamson ether synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageWilliamsonEther(BaseScoring):
    """
    Evaluates whether a Williamson ether synthesis occurs at a late stage in the synthesis route.
    Williamson ether synthesis is characterized by the formation of an ether bond (C-O-C) 
    through nucleophilic substitution of an alkyl halide by an alkoxide.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)  # Default to early stage
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Williamson ether synthesis doesn't occur
        else:
            # Late stage is better, so return complement of depth
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Detects Williamson ether synthesis by checking for:
        1. Formation of new ether bond (C-O-C)
        2. Presence of leaving group (halide or tosylate) in reactants
        3. Nucleophilic oxygen in reactants
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".") if r]
            
            if not product or not all(reactants):
                return False
            
            # Check for ether formation and Williamson conditions
            return self._is_williamson_ether_formation(product, reactants)
            
        except Exception:
            return False
    
    def _is_williamson_ether_formation(self, product, reactants) -> bool:
        """
        Checks if the reaction represents Williamson ether synthesis
        """
        # Look for new ether bonds in product
        ether_pattern = Chem.MolFromSmarts("[C]-[O]-[C]")
        if not product.HasSubstructMatch(ether_pattern):
            return False
        
        # Check for alkyl halide patterns in reactants
        halide_patterns = [
            "[C]-[Cl]",  # alkyl chloride
            "[C]-[Br]",  # alkyl bromide
            "[C]-[I]",   # alkyl iodide
            "[C]-[O]-[S](=O)(=O)-[c]"  # tosylate
        ]
        
        has_halide = False
        for reactant in reactants:
            for pattern_smarts in halide_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and reactant.HasSubstructMatch(pattern):
                    has_halide = True
                    break
            if has_halide:
                break
        
        if not has_halide:
            return False
        
        # Check for nucleophilic oxygen source (alkoxide/phenoxide)
        oxygen_nucleophile_patterns = [
            "[O-]",      # alkoxide anion
            "[OH]",      # alcohol (can be deprotonated)
            "[c]-[OH]"   # phenol
        ]
        
        has_oxygen_nucleophile = False
        for reactant in reactants:
            for pattern_smarts in oxygen_nucleophile_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and reactant.HasSubstructMatch(pattern):
                    has_oxygen_nucleophile = True
                    break
            if has_oxygen_nucleophile:
                break
        
        return has_oxygen_nucleophile
