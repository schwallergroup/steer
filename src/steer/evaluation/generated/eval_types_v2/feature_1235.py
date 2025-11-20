"""Generated evaluation code for: Late stage thiolactam formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageThiolactamFormation(BaseScoring):
    """
    Evaluates whether thiolactam formation (thionation of lactam using Lawesson's reagent)
    occurs late in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)  # Default to late stage
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Thiolactam formation doesn't occur
        else:
            # Late-stage formation is better (higher depth fraction preferred)
            if self.condition_type == "bool":
                return 10 if x >= self.target_depth else 0
            else:
                # Score based on how close to target depth
                return max(0, 10 - abs(x - self.target_depth) * 10)
    
    def hit_condition(self, d) -> bool:
        """Check if reaction involves thionation of lactam using Lawesson's reagent"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants_smiles, product_smiles = mapped_rxn.split(">>")
        reactants = reactants_smiles.split(".")
        
        # Check for Lawesson's reagent pattern
        lawessons_pattern = Chem.MolFromSmarts("[P]1(=S)[S]P(=S)[S]1")
        has_lawessons = False
        
        for reactant_smi in reactants:
            try:
                reactant_mol = Chem.MolFromSmiles(reactant_smi)
                if reactant_mol and reactant_mol.HasSubstructMatch(lawessons_pattern):
                    has_lawessons = True
                    break
            except:
                continue
                
        if not has_lawessons:
            return False
            
        # Check for lactam to thiolactam transformation
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
                
            # Pattern for thiolactam (cyclic thioamide)
            thiolactam_pattern = Chem.MolFromSmarts("[#6]1~[#6]~[#6]~[#7][C](=S)[#6]~1")
            has_thiolactam = product_mol.HasSubstructMatch(thiolactam_pattern)
            
            if not has_thiolactam:
                return False
                
            # Check if any reactant has corresponding lactam pattern
            lactam_pattern = Chem.MolFromSmarts("[#6]1~[#6]~[#6]~[#7][C](=O)[#6]~1")
            has_lactam_reactant = False
            
            for reactant_smi in reactants:
                try:
                    reactant_mol = Chem.MolFromSmiles(reactant_smi)
                    if reactant_mol and reactant_mol.HasSubstructMatch(lactam_pattern):
                        has_lactam_reactant = True
                        break
                except:
                    continue
                    
            return has_lactam_reactant
            
        except:
            return False
