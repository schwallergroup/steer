"""Generated evaluation code for: Boc protecting group strategy for amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectingGroupStrategy(BaseScoring):
    """
    Evaluates whether Boc protecting group strategy is used for amine protection.
    Checks for the presence of Boc protection reactions and their timing in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.functional_group = config["parameters"]["functional_group"]
        self.timing = config["parameters"]["timing"]  # "early", "late", or "any"
        
        # Boc protecting group patterns
        self.boc_pattern = Chem.MolFromSmarts("NC(=O)OC(C)(C)C")  # Boc-protected amine
        self.free_amine_pattern = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")  # Free primary/secondary amine
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Boc protection not found
        
        if self.timing == "early":
            return 1 - x  # Earlier is better
        elif self.timing == "late":
            return x  # Later is better
        else:  # "any"
            return 1  # Just presence is good
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves Boc protection of an amine.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
        
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for Boc protection: free amine in reactants, Boc-protected amine in products
            has_free_amine_reactant = any(mol.HasSubstructMatch(self.free_amine_pattern) for mol in reactants)
            has_boc_product = any(mol.HasSubstructMatch(self.boc_pattern) for mol in products)
            
            # Also check for Boc reagent in reactants (Boc2O or Boc-Cl)
            boc2o_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)OC(=O)OC(C)(C)C")  # Boc2O
            boc_cl_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)Cl")  # Boc-Cl
            
            has_boc_reagent = any(
                mol.HasSubstructMatch(boc2o_pattern) or mol.HasSubstructMatch(boc_cl_pattern) 
                for mol in reactants
            )
            
            # Boc protection reaction: free amine + Boc reagent -> Boc-protected amine
            return has_free_amine_reactant and has_boc_reagent and has_boc_product
            
        except Exception:
            return False
