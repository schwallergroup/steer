"""Generated evaluation code for: Convergent synthesis via amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentAmideCoupling(BaseScoring):
    """
    Evaluates convergent synthesis routes that use amide coupling to join two major fragments.
    Detects amide bond formation reactions and scores based on how early they occur in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        # SMARTS pattern for amide bond formation (C(=O)-N)
        self.amide_pattern = Chem.MolFromSmarts("[C:1](=[O:2])[N:3]")
        
    def route_scoring(self, x) -> float:
        """
        Score the route based on when amide coupling occurs.
        Earlier convergent coupling is better for synthetic efficiency.
        """
        if x < 0:
            return 0  # No amide coupling found
        else:
            # Earlier coupling (lower depth fraction) gets higher score
            return 1 - x
            
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents an amide coupling reaction.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            product_smiles, reactants_smiles = rxn_smiles.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            # Filter out None molecules and ensure we have the expected fragment count
            reactants = [r for r in reactants if r is not None]
            if len(reactants) != self.fragment_count or product is None:
                return False
                
            # Check if product contains amide bond
            if not product.HasSubstructMatch(self.amide_pattern):
                return False
                
            # Verify this is a bond formation (amide bond present in product but not in individual reactants)
            amide_in_reactants = any(r.HasSubstructMatch(self.amide_pattern) for r in reactants)
            
            # True amide coupling: product has amide bond, but individual reactants don't
            if not amide_in_reactants:
                # Additional check: ensure reactants are substantial fragments (not small molecules)
                substantial_fragments = sum(1 for r in reactants if r.GetNumAtoms() >= 5)
                return substantial_fragments >= 2
                
            return False
            
        except Exception:
            return False
