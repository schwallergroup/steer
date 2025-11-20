"""Generated evaluation code for: Late stage nitrile introduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNitrileIntroduction(BaseScoring):
    """
    Evaluates whether a nitrile group is introduced in the final step of synthesis.
    Checks for Williamson ether formation where the substrate contains a nitrile group.
    """
    
    def __init__(self, config: Dict):
        self.reaction_type = config["parameters"]["reaction_type"]
        self.substrate_pattern = config["parameters"]["substrate_contains"]
        self.timing = config["parameters"]["timing"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        elif self.timing == "final_step":
            # For final step requirement, only depth 0 (first reaction) gets full score
            if x == 0:
                return 1
            else:
                return 0
        else:
            # For other timing requirements, later is better
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves nitrile-containing substrate in Williamson ether formation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
        
        # Split reaction into product and reactants
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if this is a Williamson ether formation
            if not self._is_williamson_ether_formation(product, reactants):
                return False
            
            # Check if any reactant contains nitrile group
            nitrile_pattern = Chem.MolFromSmarts(self.substrate_pattern)
            if not nitrile_pattern:
                return False
            
            for reactant in reactants:
                if reactant.HasSubstructMatch(nitrile_pattern):
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _is_williamson_ether_formation(self, product, reactants) -> bool:
        """
        Detect Williamson ether formation by checking for ether bond formation
        between reactants that become connected in the product.
        """
        # Simple heuristic: check if product has more ether linkages than sum of reactants
        ether_pattern = Chem.MolFromSmarts("[C,c]-O-[C,c]")
        
        product_ethers = len(product.GetSubstructMatches(ether_pattern))
        reactant_ethers = sum(len(r.GetSubstructMatches(ether_pattern)) for r in reactants)
        
        # Also check for alcohol + alkyl halide pattern in reactants
        alcohol_pattern = Chem.MolFromSmarts("[C,c]-[OH]")
        halide_pattern = Chem.MolFromSmarts("[C,c]-[Cl,Br,I]")
        
        has_alcohol = any(r.HasSubstructMatch(alcohol_pattern) for r in reactants)
        has_halide = any(r.HasSubstructMatch(halide_pattern) for r in reactants)
        
        return (product_ethers > reactant_ethers) and (has_alcohol or has_halide)
