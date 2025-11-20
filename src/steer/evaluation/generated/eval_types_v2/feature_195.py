"""Generated evaluation code for: Early Suzuki coupling for biaryl core formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlySuzukiCoupling(BaseScoring):
    """
    Evaluates whether Suzuki coupling occurs early in the synthesis route.
    Checks for the presence of Suzuki-Miyaura coupling reaction patterns
    and rewards earlier occurrence in the synthetic sequence.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "early")
    
    def route_scoring(self, x) -> float:
        """
        Scoring function that rewards early Suzuki coupling.
        
        Args:
            x: Depth fraction where Suzuki coupling occurs (-1 if not found)
            
        Returns:
            Score from 0-10, with higher scores for earlier coupling
        """
        if x < 0:
            return 0  # No Suzuki coupling found
        
        if self.timing == "early":
            # Reward early coupling (lower depth fraction = higher score)
            return max(0, 10 * (1 - x))
        else:
            # For other timing preferences, could be extended
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction node represents a Suzuki coupling.
        
        Args:
            d: Reaction node dictionary containing metadata
            
        Returns:
            True if this reaction is identified as Suzuki coupling
        """
        metadata = d.get("metadata", {})
        
        # Check if reaction is explicitly labeled as Suzuki
        policy_name = metadata.get("policy_name", "").lower()
        if "suzuki" in policy_name or "miyaura" in policy_name:
            return True
        
        # Check reaction SMILES for Suzuki coupling pattern
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        if not mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            reactants = rxn_parts[1].split(".")
            product = rxn_parts[0]
            
            # Look for characteristic Suzuki patterns:
            # Boronic acid/ester + aryl halide -> biaryl
            has_boron = False
            has_halide = False
            
            for reactant in reactants:
                mol = Chem.MolFromSmiles(reactant)
                if mol is None:
                    continue
                
                # Check for boronic acid/ester patterns
                boronic_acid = Chem.MolFromSmarts("[#6]-B(-O)-O")
                boronic_ester = Chem.MolFromSmarts("[#6]-B1-O-C-C-O1")  # Pinacol ester
                boronate = Chem.MolFromSmarts("[#6]-B(-[OH,O-])-[OH,O-]")
                
                if (mol.HasSubstructMatch(boronic_acid) or 
                    mol.HasSubstructMatch(boronic_ester) or 
                    mol.HasSubstructMatch(boronate)):
                    has_boron = True
                
                # Check for aryl halide patterns
                aryl_bromide = Chem.MolFromSmarts("c-Br")
                aryl_iodide = Chem.MolFromSmarts("c-I")
                aryl_chloride = Chem.MolFromSmarts("c-Cl")
                aryl_triflate = Chem.MolFromSmarts("c-OS(=O)(=O)C(F)(F)F")
                
                if (mol.HasSubstructMatch(aryl_bromide) or 
                    mol.HasSubstructMatch(aryl_iodide) or 
                    mol.HasSubstructMatch(aryl_chloride) or
                    mol.HasSubstructMatch(aryl_triflate)):
                    has_halide = True
            
            # Check if product contains biaryl linkage
            prod_mol = Chem.MolFromSmiles(product)
            if prod_mol is None:
                return False
            
            # Biaryl pattern (aromatic carbon-carbon bond)
            biaryl_pattern = Chem.MolFromSmarts("c-c")
            has_biaryl = prod_mol.HasSubstructMatch(biaryl_pattern)
            
            return has_boron and has_halide and has_biaryl
            
        except Exception:
            return False
