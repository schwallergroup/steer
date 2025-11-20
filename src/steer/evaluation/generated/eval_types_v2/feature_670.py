"""Generated evaluation code for: Suzuki coupling for biaryl bond formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SuzukiCouplingDetector(BaseScoring):
    """
    Detects Suzuki coupling reactions for biaryl bond formation.
    
    Identifies Suzuki-Miyaura coupling reactions by detecting:
    1. Formation of biaryl C-C bonds between aromatic rings
    2. Presence of boronic acid/ester patterns in reactants
    3. Loss of boron-containing groups and halides
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0
            return abs(x - self.target_depth)
    
    def hit_condition(self, d):
        """Check if this reaction is a Suzuki coupling for biaryl formation."""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in react_smiles.split(".") if r]
            
            if not product or not reactants:
                return False
            
            # Check for biaryl formation in product
            if not self._has_biaryl_formation(product, reactants):
                return False
            
            # Check for Suzuki coupling signature
            return self._is_suzuki_coupling(reactants)
            
        except Exception:
            return False
    
    def _has_biaryl_formation(self, product, reactants):
        """Check if a new biaryl C-C bond is formed."""
        # Biaryl pattern: aromatic carbon connected to aromatic carbon
        biaryl_pattern = Chem.MolFromSmarts("[c:1]-[c:2]")
        if not product.HasSubstructMatch(biaryl_pattern):
            return False
        
        # Get all biaryl bonds in product
        product_biaryls = set()
        for match in product.GetSubstructMatches(biaryl_pattern):
            atom1_map = product.GetAtomWithIdx(match[0]).GetAtomMapNum()
            atom2_map = product.GetAtomWithIdx(match[1]).GetAtomMapNum()
            if atom1_map > 0 and atom2_map > 0:
                product_biaryls.add(tuple(sorted([atom1_map, atom2_map])))
        
        # Check if this biaryl bond exists in any reactant
        for reactant in reactants:
            if not reactant.HasSubstructMatch(biaryl_pattern):
                continue
            for match in reactant.GetSubstructMatches(biaryl_pattern):
                atom1_map = reactant.GetAtomWithIdx(match[0]).GetAtomMapNum()
                atom2_map = reactant.GetAtomWithIdx(match[1]).GetAtomMapNum()
                if atom1_map > 0 and atom2_map > 0:
                    bond_pair = tuple(sorted([atom1_map, atom2_map]))
                    if bond_pair in product_biaryls:
                        product_biaryls.remove(bond_pair)
        
        # If there are new biaryl bonds, this could be biaryl formation
        return len(product_biaryls) > 0
    
    def _is_suzuki_coupling(self, reactants):
        """Check for Suzuki coupling signature in reactants."""
        has_boronic_component = False
        has_halide_component = False
        
        # Boronic acid pattern: [c,C]-B(-O)(-O)
        boronic_acid_pattern = Chem.MolFromSmarts("[#6]-B(-[OH,O])(-[OH,O])")
        # Boronic ester pattern: [c,C]-B(-O-C)(-O-C)  
        boronic_ester_pattern = Chem.MolFromSmarts("[#6]-B(-O-[#6])(-O-[#6])")
        # Aryl halide pattern: [c]-[Cl,Br,I]
        aryl_halide_pattern = Chem.MolFromSmarts("[c]-[Cl,Br,I]")
        # Pseudohalide patterns
        triflate_pattern = Chem.MolFromSmarts("[c]-OS(=O)(=O)C(F)(F)F")
        tosylate_pattern = Chem.MolFromSmarts("[c]-OS(=O)(=O)[c]")
        
        for reactant in reactants:
            # Check for boronic acid/ester
            if (reactant.HasSubstructMatch(boronic_acid_pattern) or 
                reactant.HasSubstructMatch(boronic_ester_pattern)):
                has_boronic_component = True
            
            # Check for aryl halide or pseudohalide
            if (reactant.HasSubstructMatch(aryl_halide_pattern) or
                reactant.HasSubstructMatch(triflate_pattern) or
                reactant.HasSubstructMatch(tosylate_pattern)):
                has_halide_component = True
        
        return has_boronic_component and has_halide_component
