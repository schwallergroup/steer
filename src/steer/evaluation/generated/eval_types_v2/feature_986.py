"""Generated evaluation code for: Early cyclopropane formation via alkene cyclopropanation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyCyclopropaneFormation(BaseScoring):
    """
    Evaluates if cyclopropane formation occurs early in the synthesis route
    via diazo cyclopropanation (diazo compound addition to alkene).
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "C1CC1"
        self.timing = config["parameters"]["timing"]  # "early"
        self.formation_method = config["parameters"]["formation_method"]  # "diazo_cyclopropanation"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cyclopropane formation doesn't happen
        
        if self.timing == "early":
            return 1 - x  # Earlier formation gets higher score
        else:
            return x  # Later formation gets higher score
    
    def hit_condition(self, d):
        """Check if this reaction forms a cyclopropane via diazo cyclopropanation"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Check if cyclopropane is formed (present in product but not in reactants)
            cyclopropane_pattern = Chem.MolFromSmarts(self.ring_smarts)
            product_has_cyclopropane = product.HasSubstructMatch(cyclopropane_pattern)
            
            if not product_has_cyclopropane:
                return False
                
            # Check if any reactant already has cyclopropane
            for reactant in reactants:
                if reactant.HasSubstructMatch(cyclopropane_pattern):
                    return False  # Cyclopropane already present, not formed in this step
            
            # Check for diazo cyclopropanation pattern
            # Look for diazo compound (C=[N+]=[N-]) and alkene (C=C) in reactants
            diazo_pattern = Chem.MolFromSmarts("C=[N+]=[N-]")
            alkene_pattern = Chem.MolFromSmarts("C=C")
            
            has_diazo = False
            has_alkene = False
            
            for reactant in reactants:
                if reactant.HasSubstructMatch(diazo_pattern):
                    has_diazo = True
                if reactant.HasSubstructMatch(alkene_pattern):
                    has_alkene = True
                    
            return has_diazo and has_alkene
            
        except Exception:
            return False
