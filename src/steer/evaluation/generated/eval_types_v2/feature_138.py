"""Generated evaluation code for: Late stage piperidine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates whether piperidine ring formation occurs in the later stages of synthesis.
    Rewards routes where the specified ring is formed late in the synthesis tree.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late" or "early"
        self.direction = config["parameters"]["direction"]  # "formation" or "break"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Late-stage formation is better (lower depth fraction = higher score)
        else:  # early
            return x  # Early-stage formation is better (higher depth fraction = higher score)
    
    def hit_condition(self, d) -> bool:
        """
        Detects if the specified ring is formed (or broken) in this reaction step.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        try:
            # Parse products and reactants
            product_mols = [Chem.MolFromSmiles(products)]
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".") if r]
            
            if not all(product_mols + reactant_mols):
                return False
                
            # Check for ring presence in products vs reactants
            ring_in_products = any(mol.HasSubstructMatch(self.ring_pattern) for mol in product_mols if mol)
            ring_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactant_mols if mol)
            
            if self.direction == "formation":
                # Ring formation: present in products but not in reactants
                return ring_in_products and not ring_in_reactants
            else:  # break
                # Ring breaking: present in reactants but not in products
                return ring_in_reactants and not ring_in_products
                
        except Exception:
            return False
