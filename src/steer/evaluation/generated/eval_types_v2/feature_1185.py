"""Generated evaluation code for: Single step S-alkylation final assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SingleStepSAlkylationFinal(BaseScoring):
    """
    Evaluates whether the final step of the synthesis is a single S-alkylation reaction.
    Checks if the last reaction involves formation of a C-S bond through alkylation of a thiol.
    """
    
    def __init__(self, config: Dict):
        self.step_position = config.get("step_position", 1)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # S-alkylation doesn't happen in final step
        else:
            return 10  # Perfect score if found in final step
    
    def hit_condition(self, d):
        # Check if this is the final step (depth 1 from target)
        if d.get("depth", 0) != self.step_position:
            return False
            
        # Get the mapped reaction SMILES
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if this is an S-alkylation by looking for:
            # 1. A thiol (-SH) in reactants that becomes alkylated in product
            # 2. An alkyl halide or similar electrophile in reactants
            
            return self._is_s_alkylation_reaction(product, reactants)
            
        except Exception:
            return False
    
    def _is_s_alkylation_reaction(self, product, reactants):
        """Check if the reaction is an S-alkylation"""
        
        # SMARTS pattern for thiol group
        thiol_pattern = Chem.MolFromSmarts("[SH1]")
        
        # SMARTS patterns for common alkylating agents
        alkyl_halide_patterns = [
            Chem.MolFromSmarts("[C][Cl,Br,I]"),  # Alkyl halides
            Chem.MolFromSmarts("[C][O][S](=O)(=O)[c]"),  # Tosylates
            Chem.MolFromSmarts("[C][O][S](=O)(=O)[C]"),  # Mesylates
        ]
        
        # SMARTS pattern for newly formed C-S bond (alkyl sulfide)
        alkyl_sulfide_pattern = Chem.MolFromSmarts("[C][S][!H]")
        
        # Check if product contains alkyl sulfide
        if not product.HasSubstructMatch(alkyl_sulfide_pattern):
            return False
        
        # Check if reactants contain a thiol
        has_thiol = False
        has_electrophile = False
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(thiol_pattern):
                has_thiol = True
            
            # Check for alkylating agents
            for pattern in alkyl_halide_patterns:
                if reactant.HasSubstructMatch(pattern):
                    has_electrophile = True
                    break
        
        # Must have both thiol and electrophile in reactants
        # and alkyl sulfide in product for S-alkylation
        return has_thiol and has_electrophile
