"""Generated evaluation code for: Late stage pyridine ring reduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStagePyridineReduction(BaseScoring):
    """
    Evaluates whether pyridine ring reduction occurs at late stage in the synthesis.
    Detects when an aromatic pyridine ring is reduced to piperidine.
    """
    
    def __init__(self, config: Dict):
        self.pyridine_smarts = "c1ccncc1"  # Aromatic pyridine
        self.piperidine_smarts = "C1CCNCC1"  # Saturated piperidine
        self.timing = config["parameters"].get("timing", "late")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Pyridine reduction doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Late-stage reduction is better (higher score for higher depth)
            else:
                return x  # Early-stage reduction is better (higher score for lower depth)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves pyridine ring reduction.
        Looks for pyridine in product and piperidine in reactants.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Create pattern molecules
            pyridine_pattern = Chem.MolFromSmarts(self.pyridine_smarts)
            piperidine_pattern = Chem.MolFromSmarts(self.piperidine_smarts)
            
            if not pyridine_pattern or not piperidine_pattern:
                return False
            
            # Check if product has pyridine ring
            product_has_pyridine = product.HasSubstructMatch(pyridine_pattern)
            
            # Check if any reactant has piperidine ring (reduced form)
            reactant_has_piperidine = any(r.HasSubstructMatch(piperidine_pattern) for r in reactants)
            
            # Pyridine reduction: pyridine in product -> piperidine in reactant
            return product_has_pyridine and reactant_has_piperidine
            
        except Exception:
            return False
