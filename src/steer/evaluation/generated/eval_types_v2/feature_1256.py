"""Generated evaluation code for: Sequential Williamson ether synthesis approach"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialWilliamsonEtherSynthesis(MultiRxnCondBase):
    """
    Checks for sequential Williamson ether synthesis reactions in the route.
    Requires multiple ether formations using alkyl halide + alkoxide mechanism.
    """
    
    def __init__(self, config):
        self.min_count = config.get("min_count", 2)
        self.require_sequential = config.get("sequential", True)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        williamson_reactions = []
        
        # Find all Williamson ether synthesis reactions
        for i, rxn in enumerate(reactions):
            if self.detect_williamson_ether_synthesis(rxn):
                williamson_reactions.append(i)
        
        # Check if we have enough reactions
        if len(williamson_reactions) < self.min_count:
            return False, len(reactions)
        
        # If sequential requirement, check if reactions are in sequence
        if self.require_sequential and len(williamson_reactions) >= 2:
            # Check if reactions are consecutive in the route
            sequential_found = False
            for i in range(len(williamson_reactions) - 1):
                if williamson_reactions[i+1] - williamson_reactions[i] <= 2:  # Allow some flexibility
                    sequential_found = True
                    break
            
            condition = sequential_found
        else:
            condition = len(williamson_reactions) >= self.min_count
        
        return condition, len(reactions)
    
    def detect_williamson_ether_synthesis(self, rxn):
        """
        Detect Williamson ether synthesis by looking for:
        1. Formation of C-O-C ether bond
        2. Alkyl halide (C-X where X = Cl, Br, I) as reactant
        3. Alkoxide nucleophile pattern
        """
        try:
            prod_mol = Chem.MolFromSmiles(rxn[0])
            react_mols = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
            
            # Check for ether formation in product
            ether_pattern = Chem.MolFromSmarts("[C]-O-[C]")
            if not prod_mol.HasSubstructMatch(ether_pattern):
                return False
            
            # Look for alkyl halide in reactants
            halide_patterns = [
                Chem.MolFromSmarts("[C]-Cl"),  # Alkyl chloride
                Chem.MolFromSmarts("[C]-Br"),  # Alkyl bromide  
                Chem.MolFromSmarts("[C]-I")    # Alkyl iodide
            ]
            
            has_alkyl_halide = False
            for react_mol in react_mols:
                for pattern in halide_patterns:
                    if react_mol.HasSubstructMatch(pattern):
                        has_alkyl_halide = True
                        break
                if has_alkyl_halide:
                    break
            
            # Look for alkoxide or alcohol nucleophile
            alkoxide_patterns = [
                Chem.MolFromSmarts("[C]-[O-]"),  # Alkoxide anion
                Chem.MolFromSmarts("[C]-O"),     # Alcohol (can form alkoxide in situ)
                Chem.MolFromSmarts("[Ar]-O"),    # Phenoxide
                Chem.MolFromSmarts("[Ar]-[O-]")  # Phenoxide anion
            ]
            
            has_nucleophile = False
            for react_mol in react_mols:
                for pattern in alkoxide_patterns:
                    if react_mol.HasSubstructMatch(pattern):
                        has_nucleophile = True
                        break
                if has_nucleophile:
                    break
            
            return has_alkyl_halide and has_nucleophile
            
        except Exception:
            return False
