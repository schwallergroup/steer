"""Generated evaluation code for: Suzuki coupling for biaryl bond formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SuzukiCouplingDepth(BaseScoring):
    """
    Evaluates the depth at which a Suzuki-Miyaura cross-coupling reaction occurs.
    Detects biaryl bond formation through C(sp2)-C(sp2) coupling reactions.
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
        """Check if a reaction node represents a Suzuki-Miyaura coupling."""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            products = Chem.MolFromSmiles(rxn[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            # Check for presence of boronic acid/ester and halide patterns
            boronic_acid = Chem.MolFromSmarts("[#6]-[#5](-[#8])-[#8]")  # R-B(OH)2
            boronic_ester = Chem.MolFromSmarts("[#6]-[#5]1-[#8]-[#6]-[#6]-[#8]-1")  # Boronic ester
            aryl_halide = Chem.MolFromSmarts("c-[Br,I,Cl]")  # Aryl halide
            
            has_boron_reagent = False
            has_aryl_halide = False
            
            for reactant in reactants:
                if reactant.HasSubstructMatch(boronic_acid) or reactant.HasSubstructMatch(boronic_ester):
                    has_boron_reagent = True
                if reactant.HasSubstructMatch(aryl_halide):
                    has_aryl_halide = True
            
            # Check if both components are present
            if has_boron_reagent and has_aryl_halide:
                # Verify biaryl bond formation by checking for new C(sp2)-C(sp2) bond
                biaryl_pattern = Chem.MolFromSmarts("c-c")
                
                # Count biaryl bonds in products vs sum in reactants
                product_biaryl_count = len(products.GetSubstructMatches(biaryl_pattern))
                reactant_biaryl_count = sum(len(r.GetSubstructMatches(biaryl_pattern)) for r in reactants)
                
                return product_biaryl_count > reactant_biaryl_count
                
        except Exception:
            pass
        
        return False
