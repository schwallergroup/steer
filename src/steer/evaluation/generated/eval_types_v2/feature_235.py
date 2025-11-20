"""Generated evaluation code for: Late stage diaryl ether formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageArylEtherFormation(BaseScoring):
    """
    Evaluates whether a diaryl ether bond is formed at a late stage in the synthesis route.
    Specifically looks for Williamson ether synthesis forming c-O-c linkages.
    """
    
    def __init__(self, config: Dict):
        self.bond_smarts = config["parameters"]["bond_smarts"]  # "c-O-c"
        self.timing = config["parameters"]["timing"]  # "late" 
        self.formation_method = config["parameters"]["formation_method"]  # "williamson_ether"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Diaryl ether formation doesn't happen
        else:
            # Late-stage formation is better - penalize early formation
            if self.timing == "late":
                return 1 - x  # Higher score for reactions closer to target (lower depth fraction)
            else:
                return x  # Higher score for early reactions if that's desired
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms a diaryl ether bond via Williamson ether synthesis."""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains diaryl ether pattern
            ether_pattern = Chem.MolFromSmarts(self.bond_smarts)
            if not product.HasSubstructMatch(ether_pattern):
                return False
            
            # Check if this is likely Williamson ether synthesis
            # Look for: aryl halide + phenoxide/phenol -> diaryl ether
            has_aryl_halide = False
            has_phenol_component = False
            
            aryl_halide_pattern = Chem.MolFromSmarts("c[Cl,Br,I]")  # Aryl halide
            phenol_pattern = Chem.MolFromSmarts("c[OH]")  # Phenol
            phenoxide_pattern = Chem.MolFromSmarts("c[O-]")  # Phenoxide anion
            
            for reactant in reactants:
                if reactant.HasSubstructMatch(aryl_halide_pattern):
                    has_aryl_halide = True
                if reactant.HasSubstructMatch(phenol_pattern) or reactant.HasSubstructMatch(phenoxide_pattern):
                    has_phenol_component = True
            
            # Additional check: ensure the ether bond is newly formed
            # Count diaryl ether bonds in product vs sum in reactants
            product_ether_count = len(product.GetSubstructMatches(ether_pattern))
            reactant_ether_count = sum(len(r.GetSubstructMatches(ether_pattern)) for r in reactants)
            
            ether_bond_formed = product_ether_count > reactant_ether_count
            
            return (has_aryl_halide and has_phenol_component and ether_bond_formed)
            
        except Exception:
            return False
