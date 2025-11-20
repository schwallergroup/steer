"""Generated evaluation code for: Late stage thiazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageThiazoleFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage thiazole ring formation.
    Rewards routes where thiazole rings are formed in the latter stages of synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.formation_stage = config["parameters"]["formation_stage"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No thiazole ring formation found
        else:
            # Late stage formation gets higher score
            # x is depth fraction (0=root, 1=leaves)
            if self.formation_stage == "late":
                return (1 - x) * 10  # Later formation = higher score
            else:
                return x * 10  # Earlier formation = higher score
    
    def hit_condition(self, d) -> bool:
        """Check if thiazole ring is formed in this reaction step"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if product has thiazole ring
            product_has_thiazole = product_mol.HasSubstructMatch(self.ring_pattern)
            
            if not product_has_thiazole:
                return False
            
            # Check if any reactant already has the thiazole ring
            reactants_have_thiazole = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactant_mols)
            
            # Ring formation occurs if product has thiazole but reactants don't
            return not reactants_have_thiazole
            
        except Exception:
            return False
