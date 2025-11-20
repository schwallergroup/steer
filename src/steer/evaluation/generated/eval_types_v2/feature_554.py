"""Generated evaluation code for: Late stage Suzuki cross-coupling formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiCoupling(BaseScoring):
    """
    Evaluates whether a Suzuki cross-coupling reaction occurs in the late stage of synthesis.
    Late stage is defined as occurring within the first stage_threshold fraction of the route depth.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config["parameters"].get("stage_threshold", 0.2)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't occur
        else:
            # Late-stage reaction is better - reward reactions closer to target (depth 0)
            if x <= self.stage_threshold:
                return 10  # Maximum score for truly late-stage
            else:
                # Linearly decrease score for earlier reactions
                return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Detect Suzuki cross-coupling reaction by checking for:
        1. Boronic acid/ester reactant (B-C bond)
        2. Halide reactant (C-X where X = Br, I, Cl)
        3. Biaryl product formation
        """
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
            
            # Check for boronic acid/ester patterns in reactants
            boronic_acid_pattern = Chem.MolFromSmarts("[#6][B]([OH])[OH]")  # R-B(OH)2
            boronic_ester_pattern = Chem.MolFromSmarts("[#6][B]1OC[CH2]O1")  # Boronic ester
            pinacol_ester_pattern = Chem.MolFromSmarts("[#6][B]1OC(C)(C)C(C)(C)O1")  # Pinacol ester
            
            has_boron_reagent = any(
                reactant.HasSubstructMatch(boronic_acid_pattern) or 
                reactant.HasSubstructMatch(boronic_ester_pattern) or
                reactant.HasSubstructMatch(pinacol_ester_pattern)
                for reactant in reactants
            )
            
            # Check for aryl/vinyl halide patterns
            aryl_bromide = Chem.MolFromSmarts("c[Br]")
            aryl_iodide = Chem.MolFromSmarts("c[I]") 
            aryl_chloride = Chem.MolFromSmarts("c[Cl]")
            vinyl_halide = Chem.MolFromSmarts("C=C[Br,I,Cl]")
            
            has_halide = any(
                reactant.HasSubstructMatch(aryl_bromide) or
                reactant.HasSubstructMatch(aryl_iodide) or 
                reactant.HasSubstructMatch(aryl_chloride) or
                reactant.HasSubstructMatch(vinyl_halide)
                for reactant in reactants
            )
            
            # Check for biaryl formation in product
            biaryl_pattern = Chem.MolFromSmarts("c-c")  # Direct C-C bond between aromatics
            has_biaryl_product = product.HasSubstructMatch(biaryl_pattern)
            
            return has_boron_reagent and has_halide and has_biaryl_product
            
        except Exception:
            return False
