"""Generated evaluation code for: Claisen condensation for β-diketone formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ClaisenCondensation(BaseScoring):
    """
    Evaluates the presence and timing of Claisen condensation reactions
    that form β-diketone intermediates for pyrazole synthesis.
    
    Detects C-C bond formation between carbonyl carbon and α-carbon
    adjacent to another carbonyl group, characteristic of Claisen condensation.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition met
                return 1 if x >= 0 else 0
        else:
            if x < 0:
                return 0  # Reaction not found
            return max(0, 1 - abs(x - self.target_depth))
    
    def hit_condition(self, d):
        """Check if reaction involves Claisen condensation forming β-diketone"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains β-diketone pattern
            beta_diketone_pattern = Chem.MolFromSmarts("[#6](=[#8])-[#6]-[#6](=[#8])")
            if not product.HasSubstructMatch(beta_diketone_pattern):
                return False
            
            # Check for Claisen condensation pattern:
            # Formation of C-C bond between carbonyl carbon and activated methylene
            return self._detect_claisen_pattern(product, reactants)
            
        except Exception:
            return False
    
    def _detect_claisen_pattern(self, product, reactants):
        """Detect if the reaction matches Claisen condensation pattern"""
        # Look for carbonyl-containing reactants that could undergo Claisen condensation
        carbonyl_reactants = []
        for reactant in reactants:
            carbonyl_pattern = Chem.MolFromSmarts("[#6](=[#8])")
            if reactant.HasSubstructMatch(carbonyl_pattern):
                carbonyl_reactants.append(reactant)
        
        if len(carbonyl_reactants) < 2:
            return False
        
        # Check for presence of ester or ketone patterns in reactants
        ester_pattern = Chem.MolFromSmarts("[#6](=[#8])-[#8]-[#6]")
        ketone_pattern = Chem.MolFromSmarts("[#6]-[#6](=[#8])-[#6]")
        
        has_ester_or_ketone = any(
            r.HasSubstructMatch(ester_pattern) or r.HasSubstructMatch(ketone_pattern)
            for r in carbonyl_reactants
        )
        
        # Check for activated methylene (carbon adjacent to electron-withdrawing group)
        activated_methylene = Chem.MolFromSmarts("[#6]-[#6H2]-[#6](=[#8])")
        has_activated_methylene = any(
            r.HasSubstructMatch(activated_methylene) for r in carbonyl_reactants
        )
        
        return has_ester_or_ketone and has_activated_methylene
