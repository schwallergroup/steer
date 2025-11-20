"""Generated evaluation code for: Early chloropyridine activation via hydroxyl displacement"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ChloropyridineActivation(BaseScoring):
    """
    Evaluates routes for early chloropyridine activation via hydroxyl displacement.
    Checks for conversion of hydroxypyridine to chloropyridine using reagents like POCl3.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "early")  # "early" means lower depth is better
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't happen
        else:
            if self.timing_preference == "early":
                return 1 - x  # Early stage is better (lower depth fraction)
            else:
                return x  # Late stage is better (higher depth fraction)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves hydroxyl displacement to form chloropyridine
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        try:
            # Parse product molecule
            prod_mol = Chem.MolFromSmiles(products)
            if not prod_mol:
                return False
                
            # Parse reactant molecules
            reactant_smiles = reactants.split(".")
            reactant_mols = []
            for smi in reactant_smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    reactant_mols.append(mol)
                    
            if not reactant_mols:
                return False
                
            # Check for hydroxypyridine -> chloropyridine conversion
            return self._is_hydroxyl_to_chloro_pyridine_conversion(reactant_mols, prod_mol)
            
        except Exception:
            return False
    
    def _is_hydroxyl_to_chloro_pyridine_conversion(self, reactants, product):
        """
        Check if reaction converts hydroxypyridine to chloropyridine
        """
        # Pattern for hydroxypyridine (pyridine with OH group)
        hydroxypyridine_pattern = Chem.MolFromSmarts("[#6]1:[#6]:[#6]:[#7]:[#6]:[#6]:1-[OH1]")
        # Pattern for chloropyridine (pyridine with Cl group)  
        chloropyridine_pattern = Chem.MolFromSmarts("[#6]1:[#6]:[#6]:[#7]:[#6]:[#6]:1-[Cl]")
        
        if not hydroxypyridine_pattern or not chloropyridine_pattern:
            return False
            
        # Check if any reactant contains hydroxypyridine
        has_hydroxypyridine = any(mol.HasSubstructMatch(hydroxypyridine_pattern) for mol in reactants)
        
        # Check if product contains chloropyridine
        has_chloropyridine = product.HasSubstructMatch(chloropyridine_pattern)
        
        # Additional check for typical reagents (POCl3, etc.)
        has_chlorinating_reagent = self._has_chlorinating_reagent(reactants)
        
        return has_hydroxypyridine and has_chloropyridine and has_chlorinating_reagent
    
    def _has_chlorinating_reagent(self, reactants):
        """
        Check for presence of common chlorinating reagents like POCl3
        """
        for mol in reactants:
            mol_smiles = Chem.MolToSmiles(mol)
            # Check for POCl3 or similar phosphorus-containing chlorinating agents
            if any(pattern in mol_smiles.upper() for pattern in ["POCl3", "PCl5", "SOCl2"]):
                return True
            # Check for phosphorus and chlorine atoms together
            has_phosphorus = any(atom.GetSymbol() == 'P' for atom in mol.GetAtoms())
            has_chlorine = any(atom.GetSymbol() == 'Cl' for atom in mol.GetAtoms())
            if has_phosphorus and has_chlorine:
                return True
        return False
