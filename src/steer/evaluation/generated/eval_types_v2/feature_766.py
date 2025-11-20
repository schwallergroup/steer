"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy where multiple fragments are coupled together.
    Checks if the route involves coupling of separate fragments at a specific depth.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.target_coupling_depth = config.get("coupling_depth", 2)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        else:
            # Reward coupling at target depth, penalize deviation
            depth_penalty = abs(x - (self.target_coupling_depth / 10.0))
            return max(0, 1 - depth_penalty)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling step.
        A convergent step has multiple reactants that are combined into one product.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            reactants = reactants_smiles.split(".")
            
            # Must have at least the required number of fragments
            if len(reactants) < self.fragment_count:
                return False
                
            # Check that reactants are substantial fragments (not just small reagents)
            substantial_reactants = []
            for r_smiles in reactants:
                mol = Chem.MolFromSmiles(r_smiles)
                if mol and self._is_substantial_fragment(mol):
                    substantial_reactants.append(mol)
            
            # Must have at least the required number of substantial fragments
            if len(substantial_reactants) < self.fragment_count:
                return False
                
            # Check if this looks like a coupling reaction
            return self._is_coupling_reaction(rxn_smiles)
            
        except Exception:
            return False
    
    def _is_substantial_fragment(self, mol) -> bool:
        """
        Check if a molecule is a substantial fragment (not just a small reagent).
        Uses molecular weight and atom count as proxies.
        """
        if not mol:
            return False
            
        atom_count = mol.GetNumAtoms()
        heavy_atom_count = mol.GetNumHeavyAtoms()
        
        # Must have reasonable size to be considered a fragment
        return heavy_atom_count >= 6 and atom_count >= 8
    
    def _is_coupling_reaction(self, rxn_smiles: str) -> bool:
        """
        Check if the reaction involves typical coupling patterns like:
        - Ether formation (C-O-C)
        - C-C bond formation
        - Amide formation (C-N)
        """
        try:
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            
            if not product:
                return False
            
            # Look for common coupling patterns in the product
            coupling_patterns = [
                "[#6]-O-[#6]",  # Ether linkage (diaryl ether)
                "[#6]-[#6]",    # C-C bond
                "[#6]-N-[#6]",  # Amide/amine linkage
                "c-O-c",        # Aromatic ether
                "c-c",          # Aromatic C-C
                "[#6]=[#6]",    # C=C double bond formation
            ]
            
            for pattern in coupling_patterns:
                if product.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    return True
                    
            return False
            
        except Exception:
            return False
