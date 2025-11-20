"""Generated evaluation code for: Late stage thionation using Lawesson's reagent"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageLawessonsThionation(BaseScoring):
    """
    Evaluates whether Lawesson's reagent thionation occurs at late stages.
    Detects conversion of lactams to thiolactams using Lawesson's reagent,
    favoring reactions that occur later in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)  # Late stage preference

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Thionation doesn't happen
        else:
            # Late-stage thionation is better (higher depth fraction preferred)
            if self.condition_type == "bool":
                return 1 if x >= self.target_depth else 0
            else:
                # Reward later stage reactions more
                return min(10, x * 10)

    def hit_condition(self, d):
        """Check if reaction involves Lawesson's reagent thionation"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        # Check for Lawesson's reagent presence
        if not self._contains_lawessons_reagent(reactants):
            return False
            
        # Check for lactam to thiolactam conversion
        return self._is_thionation_reaction(products, reactants)
    
    def _contains_lawessons_reagent(self, reactants_smiles):
        """Detect Lawesson's reagent substructure"""
        # Lawesson's reagent core structure: aromatic rings with P=S units
        lawessons_patterns = [
            "P1(=S)SP(=S)(c2ccccc2)S1c1ccccc1",  # Full Lawesson's reagent
            "P(=S)(S)c1ccccc1",  # Lawesson's reagent fragment
            "[P](=S)S",  # Simplified P=S-S pattern
        ]
        
        reactant_mols = []
        for r_smi in reactants_smiles.split("."):
            mol = Chem.MolFromSmiles(r_smi)
            if mol:
                reactant_mols.append(mol)
        
        for pattern in lawessons_patterns:
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol:
                for mol in reactant_mols:
                    if mol.HasSubstructMatch(pattern_mol):
                        return True
        return False
    
    def _is_thionation_reaction(self, products_smiles, reactants_smiles):
        """Check if reaction converts C=O to C=S (thionation)"""
        # Lactam pattern (cyclic amide)
        lactam_pattern = Chem.MolFromSmarts("[C](=O)[N]")
        # Thiolactam pattern (cyclic thioamide) 
        thiolactam_pattern = Chem.MolFromSmarts("[C](=S)[N]")
        
        if not lactam_pattern or not thiolactam_pattern:
            return False
            
        # Parse molecules
        product_mols = []
        for p_smi in products_smiles.split("."):
            mol = Chem.MolFromSmiles(p_smi)
            if mol:
                product_mols.append(mol)
                
        reactant_mols = []  
        for r_smi in reactants_smiles.split("."):
            mol = Chem.MolFromSmiles(r_smi)
            if mol:
                reactant_mols.append(mol)
        
        # Check if reactants contain lactam and products contain thiolactam
        has_lactam_reactant = any(mol.HasSubstructMatch(lactam_pattern) for mol in reactant_mols)
        has_thiolactam_product = any(mol.HasSubstructMatch(thiolactam_pattern) for mol in product_mols)
        
        return has_lactam_reactant and has_thiolactam_product
