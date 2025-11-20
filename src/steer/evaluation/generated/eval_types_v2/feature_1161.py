"""Generated evaluation code for: Late stage Williamson ether formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageWilliamsonEther(BaseScoring):
    """
    Evaluates synthesis routes for late-stage Williamson ether formation.
    Detects phenol + alkyl halide reactions and rewards when they occur late in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.target_position = config.get("step_position", "final")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Williamson ether formation doesn't happen
        else:
            # Late-stage formation is better (closer to 1.0 is later)
            # Score from 1-10, with 10 being latest
            return 1 + (9 * x)
    
    def hit_condition(self, d) -> bool:
        """
        Detect Williamson ether formation by checking for:
        1. Phenol reactant (ArOH)
        2. Alkyl halide reactant (R-X where X = Cl, Br, I)
        3. Ether product formation (Ar-O-R)
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            product = Chem.MolFromSmiles(rxn[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for phenol pattern (aromatic OH)
            phenol_pattern = Chem.MolFromSmarts("[cH1,c]1[cH1,c][cH1,c][cH1,c]([OH1])[cH1,c][cH1,c]1")
            has_phenol = any(r.HasSubstructMatch(phenol_pattern) for r in reactants)
            
            # Check for alkyl halide pattern
            alkyl_halide_pattern = Chem.MolFromSmarts("[CX4][Cl,Br,I]")
            has_alkyl_halide = any(r.HasSubstructMatch(alkyl_halide_pattern) for r in reactants)
            
            # Check for ether formation in product (aromatic ether)
            aromatic_ether_pattern = Chem.MolFromSmarts("c-O-[CX4]")
            has_aromatic_ether = product.HasSubstructMatch(aromatic_ether_pattern)
            
            # Additional check: ensure we're not just detecting any ether formation
            # Verify that the phenolic OH is consumed (fewer OH groups in product vs reactants)
            reactant_oh_count = sum(len(r.GetSubstructMatches(Chem.MolFromSmarts("[OH1]"))) for r in reactants)
            product_oh_count = len(product.GetSubstructMatches(Chem.MolFromSmarts("[OH1]")))
            oh_consumed = reactant_oh_count > product_oh_count
            
            return has_phenol and has_alkyl_halide and has_aromatic_ether and oh_consumed
            
        except Exception:
            return False
