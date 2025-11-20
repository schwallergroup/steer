"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy by detecting when multiple fragments
    are coupled together at a specified stage of the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["fragment_count"]
        self.coupling_stage = config["coupling_stage"]
        self.condition_type = config.get("target_depth", {}).get("type", "value")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No convergent coupling found
        
        if self.coupling_stage == "late":
            # Reward late-stage coupling (higher depth fraction is better)
            if self.condition_type == "bool":
                return 1 if x >= 0.7 else 0
            else:
                return max(0, min(10, 10 * x))
        elif self.coupling_stage == "early":
            # Reward early-stage coupling (lower depth fraction is better)
            if self.condition_type == "bool":
                return 1 if x <= 0.3 else 0
            else:
                return max(0, min(10, 10 * (1 - x)))
        else:
            # Mid-stage coupling
            if self.condition_type == "bool":
                return 1 if 0.3 <= x <= 0.7 else 0
            else:
                target = 0.5
                return max(0, min(10, 10 * (1 - abs(x - target))))

    def hit_condition(self, d) -> bool:
        """
        Detects convergent coupling by checking if a reaction combines
        the specified number of fragments into a single product.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            prod_smiles, react_smiles = mapped_rxn.split(">>")
            reactants = react_smiles.split(".")
            
            # Check if we have the expected number of reactant fragments
            if len(reactants) != self.fragment_count:
                return False
                
            # Verify reactants are substantial fragments (not just small reagents)
            substantial_reactants = []
            for r_smiles in reactants:
                mol = Chem.MolFromSmiles(r_smiles)
                if mol and self._is_substantial_fragment(mol):
                    substantial_reactants.append(mol)
            
            # Must have exactly the target number of substantial fragments
            if len(substantial_reactants) != self.fragment_count:
                return False
                
            # Verify the product contains structural elements from multiple reactants
            prod_mol = Chem.MolFromSmiles(prod_smiles)
            if not prod_mol:
                return False
                
            # Check that the product is significantly larger than individual reactants
            prod_heavy_atoms = prod_mol.GetNumHeavyAtoms()
            total_reactant_atoms = sum(mol.GetNumHeavyAtoms() for mol in substantial_reactants)
            
            # Product should contain most atoms from reactants (allowing for small losses)
            if prod_heavy_atoms < 0.8 * total_reactant_atoms:
                return False
                
            return True
            
        except Exception:
            return False
    
    def _is_substantial_fragment(self, mol) -> bool:
        """
        Determines if a molecule is a substantial synthetic fragment
        rather than a small reagent or catalyst.
        """
        if not mol:
            return False
            
        heavy_atom_count = mol.GetNumHeavyAtoms()
        
        # Must have at least 5 heavy atoms to be considered substantial
        if heavy_atom_count < 5:
            return False
            
        # Exclude common small reagents/catalysts
        small_reagent_patterns = [
            "O",      # water
            "CO",     # carbon monoxide  
            "O=O",    # oxygen
            "[H][H]", # hydrogen
            "C=O",    # formaldehyde
            "CC=O",   # acetaldehyde
            "CCO",    # ethanol
            "CO",     # methanol
            "[Na+]",  # sodium ion
            "[K+]",   # potassium ion
            "[Cl-]",  # chloride
            "[Br-]",  # bromide
        ]
        
        mol_smiles = Chem.MolToSmiles(mol)
        if mol_smiles in small_reagent_patterns:
            return False
            
        return True
