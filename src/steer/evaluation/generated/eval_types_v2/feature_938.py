"""Generated evaluation code for: Benzoyl protection strategy for Grignard compatibility"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzoylProtectionStrategy(BaseScoring):
    """
    Evaluates synthesis routes for benzoyl protection strategy enabling Grignard reactions.
    Checks if benzoyl protection of hydroxyl groups occurs before Grignard addition reactions.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.0)
        
        # SMARTS patterns
        self.benzoyl_pattern = Chem.MolFromSmarts("[OH1][C](=O)c1ccccc1")  # Benzoyl ester
        self.hydroxyl_pattern = Chem.MolFromSmarts("[OH1][C]")  # Simple hydroxyl
        self.grignard_pattern = Chem.MolFromSmarts("[Mg][Br,Cl,I]")  # Grignard reagent
        
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
            else:
                return 1 if x >= 0 else 0
        else:
            if x < 0:
                return 0  # Protection strategy not found
            return max(0, 10 - abs(x - self.target_depth) * 10)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves benzoyl protection of hydroxyl groups
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check if reaction introduces benzoyl protection
            reactant_has_free_oh = any(mol.HasSubstructMatch(self.hydroxyl_pattern) for mol in reactants)
            product_has_benzoyl = any(mol.HasSubstructMatch(self.benzoyl_pattern) for mol in products)
            
            # Check for presence of benzoyl chloride or similar benzoyl source
            benzoyl_chloride_pattern = Chem.MolFromSmarts("[C](=O)(Cl)c1ccccc1")
            has_benzoyl_reagent = any(mol.HasSubstructMatch(benzoyl_chloride_pattern) for mol in reactants)
            
            return reactant_has_free_oh and product_has_benzoyl and has_benzoyl_reagent
            
        except Exception:
            return False
    
    def has_grignard_later(self, route_data) -> bool:
        """
        Check if Grignard reaction occurs later in the synthesis route
        """
        def traverse_route(node):
            if isinstance(node, dict):
                metadata = node.get("metadata", {})
                mapped_rxn = metadata.get("mapped_reaction_smiles", "")
                
                if mapped_rxn and ">>" in mapped_rxn:
                    try:
                        reactants_smiles = mapped_rxn.split(">>")[0]
                        reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
                        reactants = [mol for mol in reactants if mol is not None]
                        
                        if any(mol.HasSubstructMatch(self.grignard_pattern) for mol in reactants):
                            return True
                    except Exception:
                        pass
                
                # Check children
                for child in node.get("children", []):
                    if traverse_route(child):
                        return True
            return False
        
        return traverse_route(route_data)
