"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy where multiple fragments are joined.
    Checks if the synthesis route involves combining separately prepared fragments
    at a specified convergence stage.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["parameters"]["fragment_count"]
        self.convergence_stage = config["parameters"]["convergence_stage"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent step not found
        
        if self.convergence_stage == "final":
            # Reward convergence happening later in synthesis (closer to final product)
            return 1 - x
        elif self.convergence_stage == "early":
            # Reward early convergence
            return x
        else:
            # Mid-stage convergence - penalize extremes
            return 1 - abs(x - 0.5) * 2
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent step with the required number of fragments.
        A convergent reaction combines multiple non-trivial fragments (>3 heavy atoms each).
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            # Parse reaction: product >> reactants
            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                return False
                
            product_smiles = parts[0]
            reactants_smiles = parts[1].split(".")
            
            # Filter out small molecules (catalysts, reagents) - keep only fragments with >3 heavy atoms
            significant_fragments = []
            for reactant_smiles in reactants_smiles:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol is not None:
                    heavy_atom_count = mol.GetNumHeavyAtoms()
                    if heavy_atom_count > 3:  # Threshold for significant fragment
                        significant_fragments.append(mol)
            
            # Check if we have the required number of fragments
            if len(significant_fragments) >= self.fragment_count:
                # Additional check: ensure fragments are actually being coupled
                # (not just mixed with small reagents)
                product_mol = Chem.MolFromSmiles(product_smiles)
                if product_mol is not None:
                    product_heavy_atoms = product_mol.GetNumHeavyAtoms()
                    total_fragment_atoms = sum(mol.GetNumHeavyAtoms() for mol in significant_fragments)
                    
                    # Allow for some atom loss in coupling reactions (±20%)
                    atom_ratio = product_heavy_atoms / total_fragment_atoms if total_fragment_atoms > 0 else 0
                    if 0.8 <= atom_ratio <= 1.2:
                        return True
            
            return False
            
        except Exception:
            return False
