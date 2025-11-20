"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy where the target is assembled
    from multiple fragments at a specific depth in the route.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["fragment_count"]
        self.target_coupling_depth = config["coupling_depth"]
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        Lower depth (earlier convergence) gets higher score.
        """
        if x < 0:
            return 0  # Convergent step not found
        
        # Perfect score if at target depth, penalty for deviation
        if self.target_coupling_depth == 0:
            # Reward early convergence (final step)
            return 10 * (1 - x)
        else:
            # Score based on how close to target depth
            depth_diff = abs(x * 100 - self.target_coupling_depth)  # x is fraction, convert to step number
            return max(0, 10 - depth_diff)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling step
        by counting the number of significant fragments being joined.
        """
        metadata = d.get("metadata", {})
        
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        
        try:
            # Split reaction into product and reactants
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            # Parse reactants (excluding small molecules/reagents)
            reactant_mols = []
            for r_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles)
                if mol and self._is_significant_fragment(mol):
                    reactant_mols.append(mol)
            
            # Check if we have the required number of significant fragments
            return len(reactant_mols) >= self.fragment_count
            
        except Exception:
            return False
    
    def _is_significant_fragment(self, mol) -> bool:
        """
        Determine if a molecule is a significant fragment vs. small reagent.
        Criteria: >5 heavy atoms, contains carbon, not a simple inorganic.
        """
        if mol is None:
            return False
            
        heavy_atom_count = mol.GetNumHeavyAtoms()
        
        # Too small to be significant fragment
        if heavy_atom_count <= 5:
            return False
            
        # Must contain carbon to be organic fragment
        has_carbon = any(atom.GetSymbol() == 'C' for atom in mol.GetAtoms())
        if not has_carbon:
            return False
            
        # Exclude common small reagents/solvents by SMARTS patterns
        small_reagents = [
            '[OH2]',  # water
            'O',      # simple oxygen compounds
            '[Na+]', '[K+]', '[Li+]',  # metal salts
            'C(=O)O',  # simple carboxylic acids
            'CCO',     # ethanol
            'CC(=O)C', # acetone
        ]
        
        for pattern in small_reagents:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                return False
                
        return True
