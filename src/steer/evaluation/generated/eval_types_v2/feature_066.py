"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two key fragments are synthesized 
    separately and joined at a specified stage (typically final step).
    
    Checks if exactly two major fragments (non-trivial reactants) are coupled
    at the target stage to form the product.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["fragment_count"]
        self.coupling_stage = config["coupling_stage"]
        self.min_heavy_atoms = config.get("min_heavy_atoms", 5)  # Minimum size for "fragment"
        
    def route_scoring(self, x) -> float:
        """
        Score based on when convergent coupling occurs.
        x is the depth fraction where convergent coupling happens.
        """
        if x < 0:
            return 0  # No convergent coupling found
        
        if self.coupling_stage == "final":
            # Reward early convergent coupling (lower depth fraction)
            return 10 * (1 - x)
        elif self.coupling_stage == "middle":
            # Reward coupling around middle of synthesis
            optimal_depth = 0.5
            return 10 * (1 - abs(x - optimal_depth))
        else:
            # General case - earlier convergence is better
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents convergent coupling of required fragments.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            if "." not in reactants_smiles:
                return False  # Not a coupling reaction
                
            reactant_list = reactants_smiles.split(".")
            
            # Count significant fragments (non-trivial reactants)
            significant_fragments = []
            for reactant_smi in reactant_list:
                mol = Chem.MolFromSmiles(reactant_smi)
                if mol is not None:
                    heavy_atom_count = mol.GetNumHeavyAtoms()
                    # Consider as significant fragment if above minimum size
                    # and not a simple reagent (like single atoms, small molecules)
                    if heavy_atom_count >= self.min_heavy_atoms:
                        significant_fragments.append(mol)
            
            # Check if we have exactly the required number of fragments
            if len(significant_fragments) != self.fragment_count:
                return False
                
            # Additional check: ensure fragments are structurally distinct
            # (not just different conformers/tautomers of same structure)
            if self.fragment_count == 2 and len(significant_fragments) == 2:
                mol1, mol2 = significant_fragments
                # Simple structural diversity check - different molecular formulas
                formula1 = Chem.rdMolDescriptors.CalcMolFormula(mol1)
                formula2 = Chem.rdMolDescriptors.CalcMolFormula(mol2)
                
                # Also check they're not too similar in size (avoid cases where
                # one fragment is much smaller, indicating reagent rather than fragment)
                size1 = mol1.GetNumHeavyAtoms()
                size2 = mol2.GetNumHeavyAtoms()
                size_ratio = min(size1, size2) / max(size1, size2)
                
                return formula1 != formula2 and size_ratio > 0.3
                
            return True
            
        except Exception:
            return False
