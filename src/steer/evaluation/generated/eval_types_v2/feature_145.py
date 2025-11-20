"""Generated evaluation code for: Late stage C-N aryl coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageArylAmination(BaseScoring):
    """
    Evaluates if C-N aryl coupling (Buchwald-Hartwig amination) occurs late in the synthesis.
    Detects formation of C-N bonds between aromatic systems and nitrogen-containing groups.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config["parameters"].get("stage_threshold", 0.7)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No aryl amination found
        
        # Late stage is better - penalize early occurrence
        if x >= self.stage_threshold:
            return 10  # Perfect score for very late stage
        else:
            # Linear penalty for earlier occurrence
            return 10 * (x / self.stage_threshold)
    
    def hit_condition(self, d) -> bool:
        """
        Detects C-N aryl coupling by checking if:
        1. Product has Ar-N bond that's not in reactants
        2. Reactants contain separate aryl halide/triflate and amine components
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            prod_smiles, react_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in react_smiles.split(".") if r]
            
            if not product or not all(reactants):
                return False
            
            # Check for aryl-nitrogen bond formation in product
            aryl_n_bonds_prod = self._find_aryl_nitrogen_bonds(product)
            if not aryl_n_bonds_prod:
                return False
            
            # Check if these bonds exist in reactants
            aryl_n_bonds_reactants = set()
            for reactant in reactants:
                aryl_n_bonds_reactants.update(self._find_aryl_nitrogen_bonds(reactant))
            
            # Look for new aryl-N bonds formed
            new_aryl_n_bonds = aryl_n_bonds_prod - aryl_n_bonds_reactants
            if not new_aryl_n_bonds:
                return False
            
            # Verify reactants contain aryl halide/triflate and amine
            has_aryl_halide = any(self._has_aryl_halide_or_triflate(r) for r in reactants)
            has_amine = any(self._has_amine_group(r) for r in reactants)
            
            return has_aryl_halide and has_amine
            
        except Exception:
            return False
    
    def _find_aryl_nitrogen_bonds(self, mol) -> set:
        """Find aromatic carbon atoms bonded to nitrogen"""
        aryl_n_pairs = set()
        
        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            
            # Check for Ar-N bond (aromatic carbon bonded to nitrogen)
            if (begin_atom.GetSymbol() == 'C' and begin_atom.GetIsAromatic() and 
                end_atom.GetSymbol() == 'N'):
                map1 = begin_atom.GetAtomMapNum()
                map2 = end_atom.GetAtomMapNum()
                if map1 and map2:
                    aryl_n_pairs.add((min(map1, map2), max(map1, map2)))
                    
            elif (end_atom.GetSymbol() == 'C' and end_atom.GetIsAromatic() and 
                  begin_atom.GetSymbol() == 'N'):
                map1 = begin_atom.GetAtomMapNum()
                map2 = end_atom.GetAtomMapNum()
                if map1 and map2:
                    aryl_n_pairs.add((min(map1, map2), max(map1, map2)))
        
        return aryl_n_pairs
    
    def _has_aryl_halide_or_triflate(self, mol) -> bool:
        """Check for aryl halides or triflates (common coupling partners)"""
        # Aryl halides: aromatic carbon bonded to halogen
        aryl_halide_pattern = Chem.MolFromSmarts("[cH0,cH1]-[Cl,Br,I]")
        if mol.HasSubstructMatch(aryl_halide_pattern):
            return True
        
        # Aryl triflates
        triflate_pattern = Chem.MolFromSmarts("c-OS(=O)(=O)C(F)(F)F")
        if mol.HasSubstructMatch(triflate_pattern):
            return True
            
        return False
    
    def _has_amine_group(self, mol) -> bool:
        """Check for primary or secondary amine groups"""
        # Primary amine
        primary_amine = Chem.MolFromSmarts("[NH2]")
        if mol.HasSubstructMatch(primary_amine):
            return True
            
        # Secondary amine (including cyclic)
        secondary_amine = Chem.MolFromSmarts("[NH1]")
        if mol.HasSubstructMatch(secondary_amine):
            return True
            
        return False
