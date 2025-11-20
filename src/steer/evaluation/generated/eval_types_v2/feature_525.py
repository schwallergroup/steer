"""Generated evaluation code for: Early Sandmeyer bromination reaction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlySandmeyerBromination(BaseScoring):
    """
    Evaluates if a Sandmeyer bromination reaction occurs early in the synthesis route.
    Detects the conversion of aniline derivatives to aryl bromides via diazonium intermediates.
    """
    
    def __init__(self, config: Dict):
        self.step_position = config["parameters"]["step_position"]
        # SMARTS patterns for detection
        self.aniline_pattern = "[cH0:1][NH2]"  # Aniline group
        self.aryl_bromide_pattern = "[cH0:1][Br]"  # Aryl bromide
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Early stage is better - invert the depth fraction
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Checks if this reaction represents a Sandmeyer bromination.
        Looks for conversion of aniline to aryl bromide at the same aromatic position.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants_smiles = rxn[0]
            products_smiles = rxn[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
                
            # Create SMARTS patterns
            aniline_smarts = Chem.MolFromSmarts(self.aniline_pattern)
            bromide_smarts = Chem.MolFromSmarts(self.aryl_bromide_pattern)
            
            if not aniline_smarts or not bromide_smarts:
                return False
            
            # Find aniline in reactants and corresponding bromide in products
            for reactant in reactants:
                if reactant.HasSubstructMatch(aniline_smarts):
                    # Get atom mapping for the carbon attached to NH2
                    matches = reactant.GetSubstructMatches(aniline_smarts)
                    for match in matches:
                        carbon_idx = match[0]  # Carbon atom index
                        carbon_mapnum = reactant.GetAtomWithIdx(carbon_idx).GetAtomMapNum()
                        
                        if carbon_mapnum > 0:
                            # Check if this carbon now has Br in products
                            for product in products:
                                if product.HasSubstructMatch(bromide_smarts):
                                    br_matches = product.GetSubstructMatches(bromide_smarts)
                                    for br_match in br_matches:
                                        br_carbon_idx = br_match[0]
                                        br_carbon_mapnum = product.GetAtomWithIdx(br_carbon_idx).GetAtomMapNum()
                                        
                                        if br_carbon_mapnum == carbon_mapnum:
                                            return True
            
            return False
            
        except Exception:
            return False
