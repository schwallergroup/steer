"""Generated evaluation code for: Early Suzuki coupling for biphenyl core"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlySuzukiCoupling(BaseScoring):
    """
    Evaluates whether Suzuki coupling occurs early in the synthesis route.
    Suzuki coupling is identified by the formation of biphenyl structures from
    aryl halides/triflates and boronic acids/esters.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config["parameters"].get("stage_threshold", 0.3)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        
        # Early coupling is better - reward if it occurs before threshold
        if x <= self.stage_threshold:
            return 10  # Perfect score for early coupling
        else:
            # Penalize later coupling, scaling from 10 to 0
            penalty = (x - self.stage_threshold) / (1.0 - self.stage_threshold)
            return max(0, 10 * (1 - penalty))
    
    def hit_condition(self, d) -> bool:
        """
        Detect Suzuki coupling by identifying:
        1. Formation of biphenyl bond from two separate aromatic rings
        2. Presence of boron-containing reactant (boronic acid/ester)
        3. Presence of halide or triflate leaving group
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            product_smiles, reactants_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains biphenyl substructure
            biphenyl_pattern = Chem.MolFromSmarts("c1ccccc1-c2ccccc2")
            if not product.HasSubstructMatch(biphenyl_pattern):
                return False
            
            # Check for boron-containing reactant (boronic acid/ester)
            boron_patterns = [
                Chem.MolFromSmarts("[#5](-O)(-O)"),  # Boronic acid/ester
                Chem.MolFromSmarts("c[#5]"),          # Aryl boron
            ]
            
            has_boron = False
            for reactant in reactants:
                if any(reactant.HasSubstructMatch(pattern) for pattern in boron_patterns):
                    has_boron = True
                    break
            
            if not has_boron:
                return False
            
            # Check for halide or triflate leaving group
            leaving_group_patterns = [
                Chem.MolFromSmarts("c[Cl,Br,I]"),                    # Aryl halides
                Chem.MolFromSmarts("cOS(=O)(=O)C(F)(F)F"),         # Triflate
                Chem.MolFromSmarts("cOS(=O)(=O)[CH3]"),             # Tosylate
            ]
            
            has_leaving_group = False
            for reactant in reactants:
                if any(reactant.HasSubstructMatch(pattern) for pattern in leaving_group_patterns):
                    has_leaving_group = True
                    break
            
            return has_leaving_group
            
        except Exception:
            return False
