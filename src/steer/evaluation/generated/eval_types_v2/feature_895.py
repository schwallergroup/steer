"""Generated evaluation code for: Late stage aryl-alkyl cross coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageArylAlkylCrossCoupling(BaseScoring):
    """
    Evaluates whether an aryl-alkyl cross coupling reaction occurs as the final step.
    Checks for C(sp2)-C(sp3) bond formation between aryl halides and alkyl halides.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "final_step")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cross coupling doesn't happen
        elif self.timing == "final_step":
            # For final step requirement, penalize if not at depth 0
            return 1 - x if x <= 0.1 else 0
        else:
            # For general late-stage, prefer smaller depth values
            return 1 - x if x >= 0 else 0
    
    def hit_condition(self, d) -> bool:
        """Check if reaction involves aryl-alkyl cross coupling"""
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Check if we have aryl and alkyl halide reactants
            has_aryl_halide = False
            has_alkyl_halide = False
            
            for reactant in reactants:
                if self._is_aryl_halide(reactant):
                    has_aryl_halide = True
                elif self._is_alkyl_halide(reactant):
                    has_alkyl_halide = True
                    
            # Check if new C(sp2)-C(sp3) bond is formed
            if has_aryl_halide and has_alkyl_halide:
                return self._has_new_aryl_alkyl_bond(product, reactants)
                
            return False
            
        except Exception:
            return False
    
    def _is_aryl_halide(self, mol) -> bool:
        """Check if molecule contains aryl halide"""
        # Aryl halides: aromatic carbon bonded to halogen
        aryl_halide_patterns = [
            "[cH0,cH1:1][F,Cl,Br,I]",  # Aromatic carbon with halogen
        ]
        
        for pattern in aryl_halide_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                return True
        return False
    
    def _is_alkyl_halide(self, mol) -> bool:
        """Check if molecule contains alkyl halide"""
        # Alkyl halides: sp3 carbon bonded to halogen
        alkyl_halide_patterns = [
            "[CH3,CH2,CH1,CH0:1][F,Cl,Br,I]",  # sp3 carbon with halogen
            "[CX4:1][F,Cl,Br,I]",  # Tetrahedral carbon with halogen
        ]
        
        for pattern in alkyl_halide_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                return True
        return False
    
    def _has_new_aryl_alkyl_bond(self, product, reactants) -> bool:
        """Check if new C(sp2)-C(sp3) bond is formed"""
        # Look for aromatic carbon bonded to sp3 carbon in product
        aryl_alkyl_pattern = "[c:1][CX4:2]"  # aromatic C bonded to sp3 C
        
        if not product.HasSubstructMatch(Chem.MolFromSmarts(aryl_alkyl_pattern)):
            return False
            
        # Get atom map numbers for the potential new bond
        matches = product.GetSubstructMatches(Chem.MolFromSmarts(aryl_alkyl_pattern))
        
        for match in matches:
            aryl_idx, alkyl_idx = match
            aryl_atom = product.GetAtomWithIdx(aryl_idx)
            alkyl_atom = product.GetAtomWithIdx(alkyl_idx)
            
            aryl_mapnum = aryl_atom.GetAtomMapNum()
            alkyl_mapnum = alkyl_atom.GetAtomMapNum()
            
            if aryl_mapnum == 0 or alkyl_mapnum == 0:
                continue
                
            # Check if these atoms are in different reactants
            aryl_reactant = None
            alkyl_reactant = None
            
            for reactant in reactants:
                reactant_mapnums = [a.GetAtomMapNum() for a in reactant.GetAtoms()]
                if aryl_mapnum in reactant_mapnums:
                    aryl_reactant = reactant
                if alkyl_mapnum in reactant_mapnums:
                    alkyl_reactant = reactant
                    
            # If atoms are from different reactants, it's a new bond
            if (aryl_reactant and alkyl_reactant and 
                aryl_reactant != alkyl_reactant):
                return True
                
        return False
