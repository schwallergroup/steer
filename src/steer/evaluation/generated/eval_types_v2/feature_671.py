"""Generated evaluation code for: Late stage nitrile introduction via cyanation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNitrileIntroduction(BaseScoring):
    """
    Evaluates routes for late-stage nitrile introduction via cyanation.
    Detects conversion of aryl bromide ([Br]-[c]) to nitrile (C#N) and 
    scores based on how late in the synthesis this occurs.
    """
    
    def __init__(self, config: Dict):
        self.bond_smarts = config["parameters"]["bond_smarts"]  # "[Br]-[c]"
        self.replacement = config["parameters"]["replacement"]  # "C#N"
        self.timing = config["parameters"]["timing"]  # "late"
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cyanation doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Later is better, so invert depth fraction
            else:
                return x  # Earlier is better (use depth fraction as-is)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction converts aryl bromide to nitrile"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product has nitrile group
            nitrile_pattern = Chem.MolFromSmarts("C#N")
            if not product.HasSubstructMatch(nitrile_pattern):
                return False
            
            # Check if any reactant has the aryl bromide pattern
            aryl_br_pattern = Chem.MolFromSmarts(self.bond_smarts)
            has_aryl_br_reactant = any(r.HasSubstructMatch(aryl_br_pattern) for r in reactants)
            
            if not has_aryl_br_reactant:
                return False
            
            # Verify transformation: find atoms that change from Br-aryl to C#N
            # Get atom map numbers for nitrile carbons in product
            nitrile_matches = product.GetSubstructMatches(nitrile_pattern)
            product_nitrile_maps = []
            
            for match in nitrile_matches:
                carbon_atom = product.GetAtomWithIdx(match[0])  # Carbon in C#N
                if carbon_atom.GetAtomMapNum() > 0:
                    product_nitrile_maps.append(carbon_atom.GetAtomMapNum())
            
            # Check if any of these carbons were connected to Br in reactants
            for reactant in reactants:
                aryl_br_matches = reactant.GetSubstructMatches(aryl_br_pattern)
                for match in aryl_br_matches:
                    carbon_atom = reactant.GetAtomWithIdx(match[1])  # Carbon in [Br]-[c]
                    if carbon_atom.GetAtomMapNum() in product_nitrile_maps:
                        return True
                        
            return False
            
        except Exception:
            return False
