"""Generated evaluation code for: Late stage pyrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates whether a specific ring (pyrazole) is formed late in the synthesis route.
    Checks for ring formation reactions where the target ring appears in products but not reactants.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1ccnn1" for pyrazole
        self.stage = config["parameters"]["stage"]  # "late" 
        self.reaction_type = config["parameters"].get("reaction_type", "")  # "Paal-Knorr"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't occur
        else:
            # Late stage formation is better - higher depth fraction gives higher score
            if self.stage == "late":
                return x * 10  # Convert to 0-10 scale, favoring late formation
            elif self.stage == "early":
                return (1 - x) * 10  # Favor early formation
            else:
                return 5  # Neutral if stage not specified
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves formation of the target ring.
        Ring formation occurs when the ring is present in products but absent in all reactants.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse products
            product_mols = []
            for prod_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(prod_smi.strip())
                if mol:
                    product_mols.append(mol)
            
            # Parse reactants  
            reactant_mols = []
            for react_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(react_smi.strip())
                if mol:
                    reactant_mols.append(mol)
            
            # Check if ring is formed: present in products but absent in reactants
            ring_in_products = any(mol.HasSubstructMatch(self.ring_pattern) for mol in product_mols)
            ring_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactant_mols)
            
            # Ring formation occurs if ring appears in products but not in reactants
            return ring_in_products and not ring_in_reactants
            
        except Exception:
            return False
