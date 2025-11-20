"""Generated evaluation code for: Late stage cyclopropane ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCyclopropaneFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage cyclopropane ring formation.
    Detects when a cyclopropane ring (C1CC1) is formed in reactions and
    rewards routes where this occurs later in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "C1CC1"
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cyclopropane formation doesn't happen
        else:
            # For late-stage preference, higher depth fraction is better
            # Convert to 0-10 scale where late formation gets higher score
            return 10 * x
    
    def hit_condition(self, d) -> bool:
        """
        Check if cyclopropane ring formation occurs in this reaction step.
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
        if len(rxn) != 2:
            return False
            
        # Parse reactants and products
        reactants_smiles = rxn[0]
        products_smiles = rxn[1]
        
        try:
            # Count cyclopropane rings in reactants
            reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            reactant_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                               for mol in reactant_mols if mol is not None)
            
            # Count cyclopropane rings in products
            product_mols = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            product_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                              for mol in product_mols if mol is not None)
            
            # Check for ring formation (more rings in products than reactants)
            if self.direction == "formation":
                return product_rings > reactant_rings
            elif self.direction == "breaking":
                return reactant_rings > product_rings
            else:
                return product_rings != reactant_rings
                
        except Exception:
            return False
