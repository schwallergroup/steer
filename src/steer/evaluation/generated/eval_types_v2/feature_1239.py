"""Generated evaluation code for: Carbonyl to thiocarbonyl conversion using Lawesson reagent"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LawessonThionation(BaseScoring):
    """
    Evaluates synthesis routes for carbonyl to thiocarbonyl conversion using Lawesson's reagent.
    Specifically looks for thionation of lactams to form thiolactams.
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
        """Check if the reaction involves Lawesson's reagent thionation of lactam to thiolactam"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            prod_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".") if p.strip()]
            react_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".") if r.strip()]
            
            # Filter out None molecules
            prod_mols = [mol for mol in prod_mols if mol is not None]
            react_mols = [mol for mol in react_mols if mol is not None]
            
            # Check for Lawesson's reagent in reactants
            lawesson_pattern = Chem.MolFromSmarts("[P]1([S])([S])[c]2[c]([c]([c]([c]([c]2)[C])[C])[c]3[c]([P]([S])([S])[S]1)[c]([c]([c]([c]3)[C])[C])")
            has_lawesson = any(mol.HasSubstructMatch(lawesson_pattern) for mol in react_mols if mol)
            
            if not has_lawesson:
                # Also check for simplified Lawesson's reagent pattern
                lawesson_simple = Chem.MolFromSmarts("P(=S)(S)")
                has_lawesson = any(mol.HasSubstructMatch(lawesson_simple) for mol in react_mols if mol)
            
            if not has_lawesson:
                return False
            
            # Check for lactam to thiolactam conversion
            lactam_pattern = Chem.MolFromSmarts("[#6]1[#6][#6][#6][#7]([#6](=O)1)")  # 6-membered lactam
            lactam_5_pattern = Chem.MolFromSmarts("[#6]1[#6][#6][#7]([#6](=O)1)")     # 5-membered lactam
            
            thiolactam_pattern = Chem.MolFromSmarts("[#6]1[#6][#6][#6][#7]([#6](=S)1)")  # 6-membered thiolactam
            thiolactam_5_pattern = Chem.MolFromSmarts("[#6]1[#6][#6][#7]([#6](=S)1)")     # 5-membered thiolactam
            
            # Check if reactants contain lactam
            has_lactam = any(mol.HasSubstructMatch(lactam_pattern) or mol.HasSubstructMatch(lactam_5_pattern) 
                           for mol in react_mols if mol)
            
            # Check if products contain thiolactam
            has_thiolactam = any(mol.HasSubstructMatch(thiolactam_pattern) or mol.HasSubstructMatch(thiolactam_5_pattern)
                               for mol in prod_mols if mol)
            
            return has_lawesson and has_lactam and has_thiolactam
            
        except Exception:
            return False
