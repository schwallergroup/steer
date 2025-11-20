"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates whether a synthesis route employs a convergent strategy by checking
    if multiple fragments are combined at a specific depth in the route tree.
    """
    
    def __init__(self, config: Dict):
        self.final_coupling_depth = config.get("final_coupling_depth", 0)
        self.fragment_count = config.get("fragment_count", 2)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        else:
            # Perfect score if coupling happens at target depth, penalty for deviation
            depth_penalty = abs(x - self.final_coupling_depth) * 2
            return max(0, 10 - depth_penalty)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling of multiple fragments.
        A convergent step is identified by having multiple reactants that are 
        substantially different (not just simple additions like protecting groups).
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        rxn_parts = rxn_smiles.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactant_smiles = rxn_parts[1].split(".")
        
        # Need at least the specified number of fragments
        if len(reactant_smiles) < self.fragment_count:
            return False
            
        # Filter out small molecules (solvents, simple reagents)
        significant_fragments = []
        for smiles in reactant_smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol and mol.GetNumHeavyAtoms() >= 5:  # Minimum size threshold
                    significant_fragments.append(mol)
            except:
                continue
                
        if len(significant_fragments) < self.fragment_count:
            return False
            
        # Check if fragments are substantially different (convergent vs linear)
        # Compare molecular weights and structural features
        if len(significant_fragments) >= 2:
            mw_ratios = []
            for i in range(len(significant_fragments)):
                for j in range(i + 1, len(significant_fragments)):
                    mw1 = Descriptors.MolWt(significant_fragments[i])
                    mw2 = Descriptors.MolWt(significant_fragments[j])
                    ratio = min(mw1, mw2) / max(mw1, mw2) if max(mw1, mw2) > 0 else 0
                    mw_ratios.append(ratio)
            
            # If fragments have similar sizes (ratio > 0.3), likely convergent
            return any(ratio > 0.3 for ratio in mw_ratios)
            
        return False
