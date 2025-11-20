"""Generated evaluation code for: Convergent synthesis via two fragment coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two fragments are coupled
    via a base-mediated cyclization reaction at late stage.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_reaction = config.get("coupling_reaction", "base_mediated_cyclization")
        self.timing = config.get("timing", "late_stage")
        
        # Define base-mediated cyclization patterns (common ring-forming reactions)
        self.cyclization_patterns = [
            "[#6]-[#6]=[O:1]",  # Carbonyl for cyclization
            "[#7:1]-[#6]=[O]",  # Amide cyclization
            "[#6:1]-[#7]",      # C-N bond formation
            "[#6:1]=[#6]",      # Alkene cyclization
            "[#6:1]-[#8]",      # Ether cyclization
        ]
    
    def route_scoring(self, x: float) -> float:
        """Convert depth fraction to score (0-10)"""
        if x < 0:
            return 0  # Convergent coupling not found
        
        # Late-stage convergent coupling is preferred
        if self.timing == "late_stage":
            return 10 * (1 - x)  # Higher score for later coupling
        elif self.timing == "early_stage":
            return 10 * x  # Higher score for earlier coupling
        else:  # mid_stage
            return 10 * (1 - abs(x - 0.5) * 2)  # Peak at middle
    
    def hit_condition(self, d: Dict) -> bool:
        """Check if reaction represents convergent fragment coupling"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            # Filter out None molecules and small molecules (solvents, bases)
            valid_reactants = [r for r in reactants if r and r.GetNumAtoms() > 5]
            
            if len(valid_reactants) != self.fragment_count:
                return False
            
            # Check if this is a convergent coupling (two substantial fragments)
            if not self._is_convergent_coupling(valid_reactants):
                return False
            
            # Check if base-mediated cyclization occurred
            if self.coupling_reaction == "base_mediated_cyclization":
                return self._is_base_mediated_cyclization(product, valid_reactants)
            
            return True
            
        except Exception:
            return False
    
    def _is_convergent_coupling(self, reactants: List) -> bool:
        """Check if reactants represent convergent fragments (similar complexity)"""
        if len(reactants) != 2:
            return False
        
        # Check molecular complexity (atom count ratio should be reasonable)
        atom_counts = [r.GetNumAtoms() for r in reactants]
        ratio = max(atom_counts) / min(atom_counts)
        
        # Fragments should be of reasonable and similar size
        return 1.5 <= ratio <= 4.0 and min(atom_counts) >= 8
    
    def _is_base_mediated_cyclization(self, product, reactants: List) -> bool:
        """Check if a cyclization reaction occurred"""
        # Count rings in product vs reactants
        product_rings = self._count_rings(product)
        reactant_rings = sum(self._count_rings(r) for r in reactants)
        
        # New ring should be formed
        if product_rings <= reactant_rings:
            return False
        
        # Check for cyclization-prone functional groups
        return any(self._has_cyclization_pattern(r) for r in reactants)
    
    def _count_rings(self, mol) -> int:
        """Count number of rings in molecule"""
        if not mol:
            return 0
        return mol.GetRingInfo().NumRings()
    
    def _has_cyclization_pattern(self, mol) -> bool:
        """Check if molecule contains patterns prone to cyclization"""
        if not mol:
            return False
        
        for pattern_smarts in self.cyclization_patterns:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return True
        return False
