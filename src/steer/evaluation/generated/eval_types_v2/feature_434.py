"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two fragments are coupled.
    Checks if the final reaction step combines exactly two fragments to form the target.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_stage = config.get("coupling_stage", "late")
        
    def route_scoring(self, x) -> float:
        """
        Score based on how late in the synthesis the convergent coupling occurs.
        Earlier convergent steps get lower scores.
        """
        if x < 0:
            return 0  # No convergent coupling found
        
        if self.coupling_stage == "late":
            # Reward later convergent steps (closer to final product)
            return 10 * (1 - x)  # x is depth fraction, so 1-x rewards shallow depth
        else:
            # For other coupling stages, just reward presence
            return 5.0
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling of fragments.
        A convergent step has multiple reactants that combine to form fewer products.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [s.strip() for s in reactants_smiles.split(".") if s.strip()]
            products = [s.strip() for s in products_smiles.split(".") if s.strip()]
            
            # Check for convergent pattern: multiple reactants -> fewer products
            if len(reactants) < self.fragment_count:
                return False
                
            # Verify reactants are substantial fragments (not just small reagents)
            substantial_reactants = []
            for r_smiles in reactants:
                mol = Chem.MolFromSmiles(r_smiles)
                if mol and self._is_substantial_fragment(mol):
                    substantial_reactants.append(mol)
            
            # Check if we have the expected number of substantial fragments
            if len(substantial_reactants) >= self.fragment_count:
                # Verify they combine to form fewer products (convergent)
                return len(products) < len(substantial_reactants)
                
            return False
            
        except Exception:
            return False
    
    def _is_substantial_fragment(self, mol) -> bool:
        """
        Determine if a molecule is a substantial fragment vs a small reagent.
        Uses molecular weight and atom count as criteria.
        """
        if not mol:
            return False
            
        atom_count = mol.GetNumAtoms()
        heavy_atom_count = mol.GetNumHeavyAtoms()
        
        # Filter out small molecules/reagents
        if heavy_atom_count < 5:  # Less than 5 heavy atoms
            return False
            
        # Filter out common small reagents by SMARTS patterns
        small_reagent_patterns = [
            "[OH2]",  # Water
            "[NH3]",  # Ammonia  
            "O=C=O",  # CO2
            "[H][H]",  # H2
            "[Li,Na,K,Mg,Ca]",  # Simple metals
            "C(=O)O",  # Formic acid and simple carboxylic acids with <3 carbons
        ]
        
        for pattern in small_reagent_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                if heavy_atom_count < 8:  # Even smaller threshold for known reagents
                    return False
        
        return True
