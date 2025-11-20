"""Generated evaluation code for: Late thiazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateThiazoleRingFormation(BaseScoring):
    """
    Evaluates whether thiazole ring formation occurs late in the synthesis route.
    Rewards routes where thiazole rings are formed after the specified stage threshold.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1scnc1"
        self.timing = config["parameters"]["timing"]  # "late"
        self.stage_threshold = config["parameters"]["stage_threshold"]  # 0.8
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Thiazole ring formation not detected
        
        if self.timing == "late":
            # Reward later formation (higher depth fraction is better)
            if x >= self.stage_threshold:
                return 10  # Perfect score for very late formation
            else:
                # Penalize early formation
                return max(0, 10 * (x / self.stage_threshold))
        else:
            # For early timing preference (reverse scoring)
            if x <= (1 - self.stage_threshold):
                return 10
            else:
                return max(0, 10 * ((1 - x) / (1 - self.stage_threshold)))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves thiazole ring formation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            # Parse product and reactants
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1].split(".")
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles if Chem.MolFromSmiles(r) is not None]
            
            if not product_mol or not reactant_mols:
                return False
            
            # Count thiazole rings in product vs reactants
            product_thiazole_count = len(product_mol.GetSubstructMatches(self.ring_pattern))
            reactant_thiazole_count = sum(len(mol.GetSubstructMatches(self.ring_pattern)) for mol in reactant_mols)
            
            # Ring formation detected if product has more thiazole rings than reactants
            return product_thiazole_count > reactant_thiazole_count
            
        except Exception:
            return False
