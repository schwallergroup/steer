"""Generated evaluation code for: Convergent synthesis via two main fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates whether a synthesis route follows a convergent strategy by checking
    if the route assembles the target from a specified number of main fragments
    at a specific coupling depth.
    """
    
    def __init__(self, config: Dict):
        self.convergent = config["convergent"]
        self.fragment_count = config["fragment_count"]
        self.coupling_step_depth = config["coupling_step_depth"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            # Convergent strategy not found
            return 0 if self.convergent else 10
        else:
            # Convergent strategy found at depth x
            if self.convergent:
                # Reward finding convergent step at target depth
                depth_penalty = abs(x - self.coupling_step_depth) * 2
                return max(0, 10 - depth_penalty)
            else:
                # Penalize finding convergent step when not wanted
                return 0
    
    def hit_condition(self, d):
        """
        Check if this reaction represents a convergent coupling step
        by analyzing if it combines the target number of fragments.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            
            # Parse reactants
            reactant_smiles = react_smiles.split(".")
            reactants = []
            for smi in reactant_smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    reactants.append(mol)
            
            # Check if we have the expected number of main fragments
            if len(reactants) != self.fragment_count:
                return False
            
            # Parse product
            product = Chem.MolFromSmiles(prod_smiles)
            if product is None:
                return False
            
            # Check that reactants are substantial fragments (not small reagents)
            main_fragments = []
            for mol in reactants:
                # Consider a fragment "main" if it has sufficient complexity
                # (more than 6 heavy atoms and at least one ring or 8+ heavy atoms)
                heavy_atom_count = mol.GetNumHeavyAtoms()
                ring_info = mol.GetRingInfo()
                num_rings = ring_info.NumRings()
                
                if heavy_atom_count > 8 or (heavy_atom_count > 6 and num_rings > 0):
                    main_fragments.append(mol)
            
            # Check if we found the expected number of main fragments
            return len(main_fragments) == self.fragment_count
            
        except Exception:
            return False
