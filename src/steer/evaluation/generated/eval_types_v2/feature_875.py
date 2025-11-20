"""Generated evaluation code for: Convergent synthesis via protecting group fragment"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentProtectingGroupSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategies that involve preparing a protecting group 
    fragment separately and then coupling it to the main scaffold via Williamson ether synthesis.
    Checks for convergence at specified depth with Grignard chemistry for fragment preparation.
    """
    
    def __init__(self, config: Dict):
        self.target_convergence_depth = config["parameters"]["convergence_depth"]
        self.fragment_type = config["parameters"]["fragment_type"]
        self.coupling_reaction = config["parameters"]["coupling_reaction"]
        
        # Define SMARTS patterns for detection
        self.grignard_pattern = Chem.MolFromSmarts("[C,c][Mg]X")  # Grignard reagent
        self.williamson_ether_pattern = Chem.MolFromSmarts("C-O-[C,c]")  # Ether linkage
        self.protecting_group_patterns = [
            Chem.MolFromSmarts("COC(C)(C)C"),  # tert-butyl methyl ether
            Chem.MolFromSmarts("COCc1ccccc1"),  # benzyl methyl ether
            Chem.MolFromSmarts("COC(=O)C(C)(C)C"),  # tert-butyl methyl carbonate
        ]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent strategy not found
        
        # Score based on how close convergence occurs to target depth
        depth_diff = abs(x - self.target_convergence_depth / 10.0)  # x is depth fraction
        return max(0, 10 - depth_diff * 10)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents convergent coupling of a protecting group fragment
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            prod_smiles, react_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in react_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for convergence (multiple non-trivial reactants)
            significant_reactants = [r for r in reactants if r.GetNumAtoms() > 3]
            if len(significant_reactants) < 2:
                return False
            
            # Check for Williamson ether formation (ether bond in product)
            has_ether = any(product.HasSubstructMatch(pattern) for pattern in [self.williamson_ether_pattern])
            if not has_ether:
                return False
            
            # Check if one reactant contains protecting group pattern
            has_protecting_group = False
            for reactant in reactants:
                for pg_pattern in self.protecting_group_patterns:
                    if reactant.HasSubstructMatch(pg_pattern):
                        has_protecting_group = True
                        break
                if has_protecting_group:
                    break
            
            # Check for Grignard chemistry involvement (may be in previous steps)
            has_grignard_related = any(
                reactant.HasSubstructMatch(self.grignard_pattern) for reactant in reactants
            ) or self._has_grignard_precursor_pattern(reactants)
            
            return has_ether and has_protecting_group and len(significant_reactants) >= 2
            
        except Exception:
            return False
    
    def _has_grignard_precursor_pattern(self, reactants) -> bool:
        """
        Check for patterns that suggest Grignard chemistry was used in fragment preparation
        """
        # Look for alkyl halides or alcohols that could come from Grignard reactions
        grignard_precursor_patterns = [
            Chem.MolFromSmarts("[C,c][Br,Cl,I]"),  # Alkyl halides
            Chem.MolFromSmarts("[C,c]CO"),         # Primary alcohols
            Chem.MolFromSmarts("[C,c][CH](O)[C,c]"), # Secondary alcohols
        ]
        
        for reactant in reactants:
            for pattern in grignard_precursor_patterns:
                if reactant.HasSubstructMatch(pattern):
                    return True
        return False
