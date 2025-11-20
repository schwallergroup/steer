"""Generated evaluation code for: Palladium carbonylation for ester formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PalladiumCarbonylation(BaseScoring):
    """
    Detects palladium-catalyzed carbonylation reactions that form esters.
    Looks for reactions where CO is inserted into C-X bonds (typically aryl halides)
    in the presence of palladium catalyst to form ester products.
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
        # Check if reaction contains carbonylation pattern
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles:
            return False
            
        parts = rxn_smiles.split(">>")
        if len(parts) != 2:
            return False
            
        reactants = parts[0]
        products = parts[1]
        
        # Check for CO as reactant (common in carbonylation)
        if "[C-]#[O+]" in reactants or "C#O" in reactants:
            return self._check_ester_formation(reactants, products)
        
        # Alternative: check for structural pattern indicating carbonylation
        return self._detect_carbonylation_pattern(reactants, products)
    
    def _check_ester_formation(self, reactants, products):
        """Check if ester is formed in products"""
        try:
            prod_mols = [Chem.MolFromSmiles(s.strip()) for s in products.split(".")]
            ester_pattern = Chem.MolFromSmarts("[C](=O)[O][C]")
            
            for mol in prod_mols:
                if mol and mol.HasSubstructMatch(ester_pattern):
                    # Check for aryl ester (common in Pd carbonylation)
                    aryl_ester_pattern = Chem.MolFromSmarts("c[C](=O)[O][C]")
                    if mol.HasSubstructMatch(aryl_ester_pattern):
                        return True
            return False
        except:
            return False
    
    def _detect_carbonylation_pattern(self, reactants, products):
        """Detect carbonylation by structural change pattern"""
        try:
            react_mols = [Chem.MolFromSmiles(s.strip()) for s in reactants.split(".")]
            prod_mols = [Chem.MolFromSmiles(s.strip()) for s in products.split(".")]
            
            # Look for aryl halide in reactants
            aryl_halide_pattern = Chem.MolFromSmarts("c[Br,I,Cl]")
            has_aryl_halide = any(mol and mol.HasSubstructMatch(aryl_halide_pattern) 
                                for mol in react_mols if mol)
            
            # Look for ester in products
            ester_pattern = Chem.MolFromSmarts("c[C](=O)[O][C]")
            has_ester = any(mol and mol.HasSubstructMatch(ester_pattern) 
                          for mol in prod_mols if mol)
            
            return has_aryl_halide and has_ester
        except:
            return False
