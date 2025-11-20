"""Generated evaluation code for: Two-step convergent synthesis approach"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates synthesis routes for convergent strategy based on total steps and convergence type.
    A convergent synthesis involves building fragments separately and joining them later,
    as opposed to linear synthesis where steps are sequential.
    """
    
    def __init__(self, config: Dict):
        self.total_steps = config["parameters"]["total_steps"]
        self.convergence_type = config["parameters"]["convergence_type"]
        self.target_convergence_depth = self._calculate_target_depth()
    
    def _calculate_target_depth(self) -> float:
        """Calculate the expected depth for convergent step based on total steps and type"""
        if self.convergence_type == "linear_short":
            # For short linear convergent, expect convergence around middle
            return 0.5
        elif self.convergence_type == "early_convergent":
            return 0.3
        elif self.convergence_type == "late_convergent":
            return 0.7
        else:
            return 0.5  # default to middle
    
    def route_scoring(self, x) -> float:
        """
        Score based on how close the convergent step occurs to the target depth
        x is the depth fraction where convergence occurs
        """
        if x < 0:
            return 0  # No convergence found
        
        # Calculate deviation from target convergence depth
        deviation = abs(x - self.target_convergence_depth)
        
        # Convert to 0-10 score, where 0 deviation = 10 points
        score = max(0, 10 - (deviation * 20))  # Scale deviation appropriately
        return score
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent step.
        A convergent reaction typically has 2+ reactants of similar complexity.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_part = rxn_smiles.split(">>")[1]  # Get reactants side
            reactant_smiles = reactants_part.split(".")
            
            # Need at least 2 reactants for convergence
            if len(reactant_smiles) < 2:
                return False
            
            # Check if reactants have similar complexity (atom count as proxy)
            reactant_mols = []
            atom_counts = []
            
            for smi in reactant_smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    reactant_mols.append(mol)
                    atom_counts.append(mol.GetNumAtoms())
            
            if len(atom_counts) < 2:
                return False
            
            # For convergent synthesis, we expect the two largest fragments
            # to be of similar size (not one tiny reagent + one large fragment)
            atom_counts.sort(reverse=True)
            largest = atom_counts[0]
            second_largest = atom_counts[1]
            
            # Consider it convergent if the ratio is reasonable
            # (not one fragment much larger than the other)
            if largest > 0:
                size_ratio = second_largest / largest
                return size_ratio >= 0.3  # Second fragment at least 30% size of largest
            
            return False
            
        except (KeyError, IndexError, AttributeError):
            return False
