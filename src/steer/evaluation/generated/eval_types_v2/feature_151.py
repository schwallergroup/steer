"""Generated evaluation code for: Late quinolone core formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class QuinoloneCoreFormation(BaseScoring):
    """
    Evaluates when quinolone core formation occurs in a synthesis route.
    Detects formation of quinolone rings via cyclization reactions and scores
    based on timing preference (early formation is typically preferred).
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.step_position = config["parameters"]["step_position"]
        
        # Create RDKit molecule pattern for quinolone core
        self.quinolone_pattern = Chem.MolFromSmarts(self.ring_smarts)
        if self.quinolone_pattern is None:
            raise ValueError(f"Invalid SMARTS pattern: {self.ring_smarts}")

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Quinolone core formation doesn't occur
        
        if self.timing == "early":
            # Early formation preferred - higher score for smaller depth fractions
            return max(0, 10 * (1 - x))
        elif self.timing == "late":
            # Late formation preferred - higher score for larger depth fractions
            return max(0, 10 * x)
        else:
            # Specific step position
            target_fraction = self.step_position / 100.0
            deviation = abs(x - target_fraction)
            return max(0, 10 * (1 - 2 * deviation))

    def hit_condition(self, d):
        """
        Check if this reaction involves quinolone core formation by detecting
        the quinolone pattern in products but not in all reactants.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            # Parse reaction SMILES
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactants = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol is not None:
                    reactants.append(mol)
            
            products = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol is not None:
                    products.append(mol)
            
            if not reactants or not products:
                return False
            
            # Check if quinolone pattern is present in products
            quinolone_in_products = any(
                mol.HasSubstructMatch(self.quinolone_pattern) 
                for mol in products
            )
            
            if not quinolone_in_products:
                return False
            
            # Check if quinolone pattern is absent in at least one key reactant
            # (indicating ring formation rather than just functional group modification)
            quinolone_in_all_reactants = all(
                mol.HasSubstructMatch(self.quinolone_pattern) 
                for mol in reactants
            )
            
            # Ring formation occurs if quinolone is in products but not in all reactants
            return quinolone_in_products and not quinolone_in_all_reactants
            
        except Exception:
            return False
