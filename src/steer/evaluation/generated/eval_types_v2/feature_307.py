"""Generated evaluation code for: Late stage aryl-aryl bond formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageArylArylFormation(BaseScoring):
    """
    Evaluates whether an aryl-aryl bond formation (specifically Suzuki coupling) 
    occurs at late stages in the synthesis route for maximum convergence.
    """
    
    def __init__(self, config: Dict):
        self.bond_type = config["parameters"]["bond_type"]
        self.timing = config["parameters"]["timing"] 
        self.direction = config["parameters"]["direction"]
        self.reaction_type = config["parameters"]["reaction_type"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Bond formation doesn't happen
        else:
            # Later stage formation gets higher score (closer to 1.0 depth is better)
            return x  # x is depth fraction, so higher values = later stages
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves aryl-aryl bond formation via Suzuki coupling"""
        metadata = d.get("metadata", {})
        
        # Check if it's a Suzuki reaction (common patterns)
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        if not rxn_smiles:
            return False
            
        # Look for Suzuki-like patterns: aryl halide + aryl boronic acid/ester
        if not self._is_suzuki_pattern(rxn_smiles):
            return False
            
        # Check if aryl-aryl bond is being formed
        return self._detects_aryl_aryl_formation(rxn_smiles)
    
    def _is_suzuki_pattern(self, rxn_smiles: str) -> bool:
        """Detect Suzuki coupling pattern"""
        try:
            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                return False
                
            reactants = parts[1].split(".")
            
            # Look for aryl halide (Br, I, Cl on aromatic ring)
            aryl_halide_pattern = Chem.MolFromSmarts("c[Br,I,Cl]")
            has_aryl_halide = False
            
            # Look for boronic acid/ester
            boronic_patterns = [
                Chem.MolFromSmarts("cB(O)O"),  # boronic acid
                Chem.MolFromSmarts("cB1OC(C)(C)C(C)(C)O1")  # pinacol ester
            ]
            has_boronic = False
            
            for reactant_smiles in reactants:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol is None:
                    continue
                    
                if aryl_halide_pattern and mol.HasSubstructMatch(aryl_halide_pattern):
                    has_aryl_halide = True
                    
                for pattern in boronic_patterns:
                    if pattern and mol.HasSubstructMatch(pattern):
                        has_boronic = True
                        break
                        
            return has_aryl_halide and has_boronic
            
        except Exception:
            return False
    
    def _detects_aryl_aryl_formation(self, rxn_smiles: str) -> bool:
        """Check if an aryl-aryl bond is formed in this reaction"""
        try:
            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                return False
                
            product = Chem.MolFromSmiles(parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in parts[1].split(".")]
            
            if product is None or any(r is None for r in reactants):
                return False
            
            # Look for biaryl pattern in product
            biaryl_pattern = Chem.MolFromSmarts("c-c")  # aromatic C-C bond
            if not product.HasSubstructMatch(biaryl_pattern):
                return False
                
            # Check that the aryl rings were separate in reactants
            # This is a simplified check - in reality would need atom mapping
            product_aromatic_atoms = sum(1 for atom in product.GetAtoms() if atom.GetIsAromatic())
            reactant_max_aromatic = max(sum(1 for atom in r.GetAtoms() if atom.GetIsAromatic()) for r in reactants)
            
            # If product has more aromatic connectivity than largest reactant, likely biaryl formation
            return product_aromatic_atoms > reactant_max_aromatic
            
        except Exception:
            return False
