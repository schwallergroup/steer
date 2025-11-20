"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by checking if the specified number of 
    fragments are joined at a particular coupling step depth.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.target_coupling_step = config.get("coupling_step", 1)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        else:
            # Earlier coupling (lower depth) is better for convergent synthesis
            depth_penalty = abs(x - (self.target_coupling_step / 10.0))
            return max(0, 1 - depth_penalty)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling of the specified
        number of fragments by analyzing the reaction SMILES.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            # Split reaction into products and reactants
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            products = rxn_parts[0].strip()
            reactants = rxn_parts[1].strip()
            
            # Count reactant fragments
            reactant_smiles = [r.strip() for r in reactants.split(".") if r.strip()]
            
            # Check if we have the expected number of fragments
            if len(reactant_smiles) != self.fragment_count:
                return False
            
            # Verify each reactant is a substantial fragment (not just small reagents)
            substantial_fragments = 0
            for reactant_smi in reactant_smiles:
                try:
                    mol = Chem.MolFromSmiles(reactant_smi)
                    if mol and self._is_substantial_fragment(mol):
                        substantial_fragments += 1
                except:
                    continue
            
            # Check if we have convergent coupling (multiple substantial fragments)
            return substantial_fragments >= self.fragment_count
            
        except Exception:
            return False
    
    def _is_substantial_fragment(self, mol) -> bool:
        """
        Determine if a molecule is a substantial synthetic fragment rather than
        a small reagent or leaving group.
        """
        if not mol:
            return False
        
        # Count heavy atoms (non-hydrogen)
        heavy_atom_count = mol.GetNumHeavyAtoms()
        
        # Fragment should have reasonable size (at least 5 heavy atoms)
        if heavy_atom_count < 5:
            return False
        
        # Check for common small reagents/leaving groups to exclude
        small_reagent_patterns = [
            "[Cl,Br,I]",  # Simple halides
            "C(=O)O",     # Carboxylic acids
            "S(=O)(=O)O", # Sulfonic acids
            "[Li,Na,K]",  # Alkali metals
            "B(O)(O)",    # Boronic acids (simple)
        ]
        
        for pattern in small_reagent_patterns:
            try:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) and heavy_atom_count < 8:
                    return False
            except:
                continue
        
        return True
