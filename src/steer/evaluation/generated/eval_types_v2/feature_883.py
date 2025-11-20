"""Generated evaluation code for: Schmidt glycosylation for C-O bond formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SchmidtGlycosylation(BaseScoring):
    """
    Evaluates the presence of Schmidt glycosylation reactions using trichloroacetimidate donors
    for C-O bond formation at a specific depth in the synthesis route.
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
        """Check if reaction involves Schmidt glycosylation with trichloroacetimidate donor"""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            products = [Chem.MolFromSmiles(p) for p in rxn[0].split(".")]
            
            # Check for trichloroacetimidate donor pattern
            trichloroacetimidate_pattern = Chem.MolFromSmarts("[C]([NH][C](=[N])[CH2]Cl)([Cl])([Cl])")
            
            # Alternative pattern for trichloroacetimidate group
            acetimidate_pattern = Chem.MolFromSmarts("[O][C](=[N])[CH2]Cl")
            
            # Check if any reactant contains trichloroacetimidate donor
            has_donor = False
            for reactant in reactants:
                if reactant and (reactant.HasSubstructMatch(trichloroacetimidate_pattern) or 
                               reactant.HasSubstructMatch(acetimidate_pattern)):
                    has_donor = True
                    break
            
            if not has_donor:
                return False
            
            # Check for C-O glycosidic bond formation
            # Look for new ether linkage between sugar-like structures
            sugar_pattern = Chem.MolFromSmarts("[CH1,CH2][O][CH1]([CH1,CH2][O])[CH1,CH2]")
            glycosidic_pattern = Chem.MolFromSmarts("[CH1][O][CH1,CH2]")
            
            # Check if products contain glycosidic bonds that weren't in reactants
            product_has_glycosidic = any(prod.HasSubstructMatch(sugar_pattern) or 
                                       prod.HasSubstructMatch(glycosidic_pattern) 
                                       for prod in products if prod)
            
            reactant_has_glycosidic = any(react.HasSubstructMatch(sugar_pattern) or 
                                        react.HasSubstructMatch(glycosidic_pattern) 
                                        for react in reactants if react)
            
            # Schmidt glycosylation should form new glycosidic bonds
            return has_donor and product_has_glycosidic and not reactant_has_glycosidic
            
        except Exception:
            return False
