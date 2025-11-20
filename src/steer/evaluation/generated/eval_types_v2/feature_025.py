"""Generated evaluation code for: Late stage Suzuki coupling for biaryl formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiCoupling(BaseScoring):
    """
    Evaluates synthesis routes for late-stage Suzuki coupling reactions that form biaryl bonds.
    Rewards routes where Suzuki coupling occurs closer to the final product (late stage).
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.1)  # Close to final step
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No Suzuki coupling found
        else:
            # Reward late-stage coupling (low depth fraction)
            # Perfect score when x approaches 0 (very late stage)
            return max(0, 1 - x * 5)  # Scale so depth > 0.2 gets 0 score
    
    def hit_condition(self, d) -> bool:
        """
        Detects Suzuki coupling by looking for:
        1. Formation of biaryl C-C bond
        2. Presence of boronic acid/ester reactant
        3. Presence of aryl halide reactant
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        try:
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for biaryl formation in product
            biaryl_pattern = Chem.MolFromSmarts("c1ccccc1-c2ccccc2")  # Simple biaryl
            if not product.HasSubstructMatch(biaryl_pattern):
                return False
            
            # Check for boronic acid/ester reactant
            boronic_acid = Chem.MolFromSmarts("[cH0:1]B(O)O")
            boronic_ester = Chem.MolFromSmarts("[cH0:1]B1OCCO1")  # Pinacol ester
            boronic_ester2 = Chem.MolFromSmarts("[cH0:1]B(OC)OC")  # Dimethyl ester
            
            has_boron = False
            for reactant in reactants:
                if (reactant.HasSubstructMatch(boronic_acid) or 
                    reactant.HasSubstructMatch(boronic_ester) or
                    reactant.HasSubstructMatch(boronic_ester2)):
                    has_boron = True
                    break
            
            # Check for aryl halide reactant
            aryl_halide = Chem.MolFromSmarts("c[Cl,Br,I]")
            has_halide = False
            for reactant in reactants:
                if reactant.HasSubstructMatch(aryl_halide):
                    has_halide = True
                    break
            
            return has_boron and has_halide
            
        except Exception:
            return False
