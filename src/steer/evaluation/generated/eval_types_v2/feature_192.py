"""Generated evaluation code for: Late stage nitro group reduction to amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNitroReduction(BaseScoring):
    """
    Evaluates whether nitro group reduction to amine occurs at late stage in synthesis.
    Rewards routes where nitro reduction happens after the step_threshold fraction of total steps.
    """
    
    def __init__(self, config: Dict):
        self.step_threshold = config["parameters"].get("step_threshold", 0.8)
        # SMARTS patterns for nitro group reduction
        self.nitro_pattern = Chem.MolFromSmarts("[N+](=O)[O-]")  # Nitro group
        self.amine_pattern = Chem.MolFromSmarts("[NH2,NH1,NH0]")  # Amine group
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        Reward late-stage nitro reduction (high depth fraction).
        """
        if x < 0:
            return 0  # Nitro reduction doesn't happen
        
        if x >= self.step_threshold:
            return 10 * (x - self.step_threshold) / (1 - self.step_threshold)
        else:
            return 0  # Too early in synthesis
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents nitro group reduction to amine.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi)
                if mol is not None:
                    reactants.append(mol)
            
            products = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi)
                if mol is not None:
                    products.append(mol)
            
            # Check for nitro group in reactants and amine in products
            has_nitro_reactant = any(mol.HasSubstructMatch(self.nitro_pattern) for mol in reactants)
            has_amine_product = any(mol.HasSubstructMatch(self.amine_pattern) for mol in products)
            
            # Additional check: ensure nitro group is actually being reduced
            # Count nitro groups in reactants vs products
            if has_nitro_reactant and has_amine_product:
                reactant_nitro_count = sum(len(mol.GetSubstructMatches(self.nitro_pattern)) for mol in reactants)
                product_nitro_count = sum(len(mol.GetSubstructMatches(self.nitro_pattern)) for mol in products)
                
                # Nitro reduction should decrease nitro count
                return reactant_nitro_count > product_nitro_count
            
            return False
            
        except Exception:
            return False
