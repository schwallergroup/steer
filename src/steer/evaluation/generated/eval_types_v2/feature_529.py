"""Generated evaluation code for: Linear piperazine ring construction on scaffold"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LinearPiperazineConstruction(BaseScoring):
    """
    Evaluates synthesis routes for linear piperazine ring construction on scaffold.
    Checks if piperazine rings are built step-by-step on the scaffold rather than 
    through convergent coupling of pre-formed fragments.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config.get("ring_smarts", "N1CCNCC1")
        self.piperazine_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        """
        Scoring function that rewards early linear construction.
        x is the depth fraction where linear construction occurs.
        """
        if x < 0:
            return 0  # Linear construction doesn't happen
        else:
            # Earlier construction (lower x) gets higher score
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Checks if a reaction represents linear piperazine construction.
        Returns True if the reaction shows step-by-step ring building.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains piperazine
            if not product.HasSubstructMatch(self.piperazine_pattern):
                return False
            
            # For linear construction, we expect:
            # 1. Product has complete piperazine ring
            # 2. At least one reactant has partial piperazine structure (incomplete ring)
            # 3. No reactant has complete piperazine ring (not convergent)
            
            has_complete_piperazine_reactant = any(r.HasSubstructMatch(self.piperazine_pattern) for r in reactants)
            
            # If any reactant already has complete piperazine, this is convergent, not linear
            if has_complete_piperazine_reactant:
                return False
            
            # Check for partial piperazine structures that indicate linear construction
            partial_patterns = [
                "NCCN",  # Linear chain that could form piperazine
                "N1CCN",  # Partial ring
                "NCC1",   # Another partial pattern
            ]
            
            has_partial_structure = False
            for pattern_smarts in partial_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if any(r.HasSubstructMatch(pattern) for r in reactants):
                    has_partial_structure = True
                    break
            
            # Linear construction: product has complete ring, reactants have partial structures
            return has_partial_structure
            
        except Exception:
            return False
