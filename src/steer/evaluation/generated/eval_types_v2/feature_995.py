"""Generated evaluation code for: Convergent synthesis via ketone and hydrazine fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentKetoneHydrazineSynthesis(BaseScoring):
    """
    Evaluates synthesis routes for convergent strategy using ketone and hydrazine fragments
    coupled via Fischer indole synthesis. Returns better scores for routes where the 
    Fischer indole coupling occurs later in the synthesis (more convergent).
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.fragment_types = config.get("fragment_types", ["ketone", "hydrazine"])
        self.coupling_reaction = config.get("coupling_reaction", "Fischer indole synthesis")
        
        # SMARTS patterns for detecting fragments
        self.ketone_pattern = Chem.MolFromSmarts("[C](=O)")
        self.hydrazine_pattern = Chem.MolFromSmarts("[N]-[N]")
        self.indole_product_pattern = Chem.MolFromSmarts("c1ccc2[nH]ccc2c1")
    
    def route_scoring(self, x) -> float:
        """
        Converts depth fraction to score. Later coupling (higher x) gets better score.
        """
        if x < 0:
            return 0  # Fischer indole coupling doesn't occur
        else:
            # Later stage coupling is more convergent, so higher x gives better score
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Checks if this reaction represents a Fischer indole synthesis between
        ketone and hydrazine fragments.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            product = Chem.MolFromSmiles(product_smiles.strip())
            
            if not all(reactants) or not product:
                return False
            
            # Check if product contains indole scaffold
            if not product.HasSubstructMatch(self.indole_product_pattern):
                return False
            
            # Check if we have the required fragments among reactants
            has_ketone = False
            has_hydrazine = False
            
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.ketone_pattern):
                    has_ketone = True
                if reactant.HasSubstructMatch(self.hydrazine_pattern):
                    has_hydrazine = True
            
            # Require both ketone and hydrazine fragments
            return has_ketone and has_hydrazine and len(reactants) >= self.fragment_count
            
        except Exception:
            return False
