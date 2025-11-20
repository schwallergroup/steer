"""Generated evaluation code for: Late stage piperidine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates synthesis routes based on when a specific ring is formed during the synthesis.
    Rewards late-stage ring formation by checking at what depth the target ring appears
    in the product but not in the reactants (indicating ring formation).
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["ring_smarts"]
        self.timing = config.get("timing", "late")  # "early" or "late"
        self.direction = config.get("direction", "formation")  # "formation" or "break"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation/break doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later is better (closer to 1.0 for late stages)
        else:  # early
            return x  # Earlier is better (closer to 1.0 for early stages)
    
    def hit_condition(self, d) -> bool:
        """Check if the target ring is formed or broken in this reaction step."""
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        # Parse product
        product_mol = Chem.MolFromSmiles(product_smiles)
        if product_mol is None:
            return False
            
        # Parse reactants
        reactant_mols = []
        for r_smiles in reactant_smiles.split("."):
            r_mol = Chem.MolFromSmiles(r_smiles.strip())
            if r_mol is not None:
                reactant_mols.append(r_mol)
        
        if not reactant_mols:
            return False
            
        # Check ring presence in product and reactants
        ring_in_product = product_mol.HasSubstructMatch(self.ring_pattern)
        ring_in_reactants = any(r_mol.HasSubstructMatch(self.ring_pattern) for r_mol in reactant_mols)
        
        if self.direction == "formation":
            # Ring formation: ring present in product but not in any reactant
            return ring_in_product and not ring_in_reactants
        else:  # direction == "break"
            # Ring breaking: ring present in reactants but not in product
            return ring_in_reactants and not ring_in_product
