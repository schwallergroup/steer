"""Generated evaluation code for: Isoxazole ring opening for beta-keto nitrile formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class IsoxazoleRingOpening(BaseScoring):
    """
    Evaluates synthesis routes for isoxazole ring opening reactions that reveal beta-keto nitrile functionality.
    The fused isoxazole serves as a masked beta-keto nitrile precursor.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1cnoc1"
        self.timing = config["parameters"]["timing"]  # "mid"
        self.direction = config["parameters"]["direction"]  # "break"
        
        # Convert timing preference to target depth fraction
        timing_map = {"early": 0.2, "mid": 0.5, "late": 0.8}
        self.target_depth_fraction = timing_map.get(self.timing, 0.5)

    def route_scoring(self, depth_fraction) -> float:
        """Convert depth fraction to 0-10 score based on timing preference"""
        if depth_fraction < 0:  # Ring opening not found
            return 0
        
        # Score based on how close the actual depth is to target timing
        deviation = abs(depth_fraction - self.target_depth_fraction)
        # Convert to 0-10 scale, with 10 being perfect timing
        score = max(0, 10 * (1 - 2 * deviation))
        return score

    def hit_condition(self, d) -> bool:
        """Check if this reaction node performs isoxazole ring opening"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            product_smiles, reactant_smiles = rxn_smiles.split(">>")
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            if product_mol is None:
                return False
            
            # Check if product contains isoxazole ring
            isoxazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if not product_mol.HasSubstructMatch(isoxazole_pattern):
                return False
            
            # Check if any reactant has the ring opened (no longer contains isoxazole)
            reactant_mols = []
            for reactant in reactant_smiles.split("."):
                mol = Chem.MolFromSmiles(reactant)
                if mol is not None:
                    reactant_mols.append(mol)
            
            # Ring opening occurs if at least one reactant lacks the isoxazole ring
            # but we can trace the carbon/nitrogen skeleton suggesting ring opening occurred
            ring_opened = any(not mol.HasSubstructMatch(isoxazole_pattern) for mol in reactant_mols)
            
            # Additional check: look for beta-keto nitrile formation
            # Pattern for beta-keto nitrile: C(=O)CC#N or similar
            beta_keto_nitrile_pattern = Chem.MolFromSmarts("[C](=O)[CH2][C]#[N]")
            nitrile_formed = any(mol.HasSubstructMatch(beta_keto_nitrile_pattern) for mol in reactant_mols)
            
            return ring_opened and nitrile_formed
            
        except (KeyError, ValueError, AttributeError):
            return False
