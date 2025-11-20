"""Generated evaluation code for: Late imidazopyridine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ImidazopyridineRingFormation(BaseScoring):
    """
    Evaluates the timing of imidazopyridine ring formation in synthesis routes.
    Checks when the specified imidazopyridine ring system is formed and scores
    based on whether it occurs late in the synthesis as desired.
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
            return 1 - x  # Late-stage formation gets higher score
        else:  # early timing
            return x  # Early-stage formation gets higher score
    
    def hit_condition(self, d) -> bool:
        """
        Checks if the imidazopyridine ring is formed/broken in this reaction step.
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
        
        # Check for ring presence in reactants and products
        ring_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
        ring_in_products = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
        
        if self.direction == "formation":
            # Ring formation: absent in reactants, present in products
            return not ring_in_reactants and ring_in_products
        else:  # direction == "break"
            # Ring breaking: present in reactants, absent in products
            return ring_in_reactants and not ring_in_products
