"""Generated evaluation code for: Convergent synthesis via two major fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by detecting when two major fragments
    are coupled together. Looks for reactions where multiple substantial fragments
    (non-trivial reactants) combine to form the product.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_stage = config.get("coupling_stage", "late")
        self.min_heavy_atoms = config.get("min_heavy_atoms", 8)  # Minimum size for "major" fragment
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No convergent coupling found
        
        if self.coupling_stage == "late":
            # Reward late-stage convergent coupling (higher depth fraction is better)
            return x * 10
        elif self.coupling_stage == "early":
            # Reward early-stage convergent coupling (lower depth fraction is better)
            return (1 - x) * 10
        else:  # "any"
            # Just reward finding convergent coupling regardless of timing
            return 8.0
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling of major fragments.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            product_smiles, reactants_smiles = mapped_rxn.split(">>")
            reactant_list = reactants_smiles.split(".")
            
            # Need at least the specified number of reactants
            if len(reactant_list) < self.fragment_count:
                return False
            
            # Parse molecules
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactant_list]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Count "major" fragments (non-trivial reactants)
            major_fragments = []
            for mol in reactant_mols:
                heavy_atom_count = mol.GetNumHeavyAtoms()
                if heavy_atom_count >= self.min_heavy_atoms:
                    major_fragments.append(mol)
            
            # Check if we have the required number of major fragments
            if len(major_fragments) < self.fragment_count:
                return False
            
            # Additional check: ensure fragments are actually coupling
            # (product should be significantly larger than individual fragments)
            product_heavy_atoms = product_mol.GetNumHeavyAtoms()
            total_reactant_heavy_atoms = sum(mol.GetNumHeavyAtoms() for mol in major_fragments)
            
            # Allow for some atoms to be lost/gained in coupling (±3 atoms tolerance)
            if abs(product_heavy_atoms - total_reactant_heavy_atoms) > 3:
                return False
            
            return True
            
        except Exception:
            return False
