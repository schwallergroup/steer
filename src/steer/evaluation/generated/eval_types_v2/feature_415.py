"""Generated evaluation code for: Late stage tetrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageTetraRingFormation(BaseScoring):
    """
    Evaluates whether tetrazole ring formation occurs in the late stages of synthesis.
    Rewards routes where tetrazole rings are formed after the specified position threshold
    to avoid protection group issues and synthetic complications.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.position_threshold = config["parameters"]["position_threshold"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            if x >= self.position_threshold:
                return 10  # Perfect score for late-stage formation
            else:
                # Penalize early formation, scale from 0-5 based on how early
                return 5 * (x / self.position_threshold)
        else:
            # For early timing preference (reverse scoring)
            if x <= (1 - self.position_threshold):
                return 10
            else:
                return 5 * ((1 - x) / self.position_threshold)
    
    def hit_condition(self, d) -> bool:
        """
        Check if tetrazole ring is formed in this reaction step.
        """
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        # Parse reactants and products
        reactants = []
        for r_smiles in reactants_smiles.split("."):
            mol = Chem.MolFromSmiles(r_smiles)
            if mol is not None:
                reactants.append(mol)
        
        products = []
        for p_smiles in products_smiles.split("."):
            mol = Chem.MolFromSmiles(p_smiles)
            if mol is not None:
                products.append(mol)
        
        # Count tetrazole rings in reactants and products
        reactant_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                           for mol in reactants)
        product_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                          for mol in products)
        
        # Ring formation occurred if products have more tetrazole rings than reactants
        return product_rings > reactant_rings
