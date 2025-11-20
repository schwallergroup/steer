"""Generated evaluation code for: Early methyl ether protecting group installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyMethylEtherProtection(BaseScoring):
    """
    Evaluates if methyl ether protection of phenol occurs early in the synthesis route.
    Returns higher scores when the protection happens in early stages.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)  # Early stage target
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection doesn't happen
        else:
            # Early protection (low x) gets higher score
            if self.condition_type == "bool":
                return 1 if x <= self.target_depth else 0
            else:
                # Penalize late protection more heavily
                if x <= self.target_depth:
                    return 1.0
                else:
                    return max(0, 1.0 - 2 * (x - self.target_depth))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves methyl ether protection of a phenol.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Define SMARTS patterns
            phenol_pattern = Chem.MolFromSmarts("[OH1]c1ccccc1")  # Simple phenol
            methyl_ether_pattern = Chem.MolFromSmarts("[CH3]Oc1ccccc1")  # Methyl phenyl ether
            methylating_agent_pattern = Chem.MolFromSmarts("[CH3][I,Br,Cl]")  # Common methylating agents
            
            # Check if reactants contain phenol
            has_phenol_reactant = any(mol.HasSubstructMatch(phenol_pattern) for mol in reactants)
            
            # Check if reactants contain methylating agent
            has_methylating_agent = any(mol.HasSubstructMatch(methylating_agent_pattern) for mol in reactants)
            
            # Check if products contain methyl ether
            has_methyl_ether_product = any(mol.HasSubstructMatch(methyl_ether_pattern) for mol in products)
            
            # Condition: phenol + methylating agent -> methyl ether
            return has_phenol_reactant and has_methylating_agent and has_methyl_ether_product
            
        except Exception:
            return False
