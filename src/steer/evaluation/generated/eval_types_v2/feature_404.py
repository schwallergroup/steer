"""Generated evaluation code for: Early Suzuki coupling for biaryl core"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlySuzukiCoupling(BaseScoring):
    """
    Evaluates whether Suzuki-Miyaura coupling occurs early in the synthesis route
    to establish a biaryl core. Returns higher scores when the coupling happens
    within the early threshold of the route.
    """
    
    def __init__(self, config: Dict):
        self.step_threshold = config["parameters"].get("step_threshold", 0.3)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        
        # Early coupling (depth fraction < threshold) gets higher score
        if x <= self.step_threshold:
            return 10 - (x / self.step_threshold) * 5  # Score 5-10
        else:
            # Late coupling gets lower score
            late_fraction = (x - self.step_threshold) / (1.0 - self.step_threshold)
            return 5 * (1 - late_fraction)  # Score 0-5
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Suzuki-Miyaura coupling forming biaryl"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            product = Chem.MolFromSmiles(product_smiles.strip())
            
            if not all(reactants) or not product:
                return False
            
            # Check for Suzuki coupling pattern: organoborane + aryl halide -> biaryl
            has_organoborane = False
            has_aryl_halide = False
            
            # Organoborane patterns (boronic acid, boronic ester)
            borane_patterns = [
                "[#6][B]([OH])[OH]",  # Boronic acid
                "[#6][B]1OC(C)(C)C(C)(C)O1",  # Pinacol boronic ester
                "[#6][B]([OR])[OR]",  # General boronic ester
            ]
            
            # Aryl halide patterns
            halide_patterns = [
                "c[Cl,Br,I]",  # Aromatic halide
                "[cH0][Cl,Br,I]",  # Substituted aromatic halide
            ]
            
            for reactant in reactants:
                # Check for organoborane
                for pattern in borane_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_organoborane = True
                        break
                
                # Check for aryl halide
                for pattern in halide_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_aryl_halide = True
                        break
            
            # Check if product contains biaryl core (two connected aromatic rings)
            biaryl_pattern = "c1ccccc1-c2ccccc2"  # Simple biaryl
            has_biaryl_product = product.HasSubstructMatch(Chem.MolFromSmarts(biaryl_pattern))
            
            return has_organoborane and has_aryl_halide and has_biaryl_product
            
        except Exception:
            return False
