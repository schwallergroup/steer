"""Generated evaluation code for: Early stage Boc protection strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectionStrategy(BaseScoring):
    """
    Evaluates whether Boc protection of secondary amines occurs at early stages.
    Checks for the introduction of Boc protecting groups on nitrogen atoms at a specific step.
    """
    
    def __init__(self, config: Dict):
        self.target_step = config["parameters"]["step_number"]
        self.timing = config["parameters"]["timing"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Boc protection doesn't happen
        
        if self.timing == "early":
            # Early protection is better - penalize later steps
            if x <= 0.2:  # Within first 20% of synthesis
                return 1.0
            else:
                return max(0, 1.0 - (x - 0.2) * 2.5)
        else:
            return 1 - x  # General case: earlier is better
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves Boc protection of a secondary amine"""
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
        if len(rxn) != 2:
            return False
            
        reactants = rxn[0]
        products = rxn[1]
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Check for Boc group introduction
            boc_pattern = Chem.MolFromSmarts("[NH1][C](=O)[O][C](C)(C)C")  # Boc-protected amine
            secondary_amine_pattern = Chem.MolFromSmarts("[NH1]")  # Secondary amine
            
            # Check if reactants contain secondary amine but no Boc
            has_secondary_amine_reactant = any(mol.HasSubstructMatch(secondary_amine_pattern) 
                                             for mol in reactant_mols)
            has_boc_reactant = any(mol.HasSubstructMatch(boc_pattern) 
                                 for mol in reactant_mols)
            
            # Check if products contain Boc-protected amine
            has_boc_product = any(mol.HasSubstructMatch(boc_pattern) 
                                for mol in product_mols)
            
            # Boc protection occurs if:
            # 1. Reactants have secondary amine but no Boc group
            # 2. Products have Boc-protected amine
            return (has_secondary_amine_reactant and 
                   not has_boc_reactant and 
                   has_boc_product)
                   
        except Exception:
            return False
