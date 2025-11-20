"""Generated evaluation code for: Boc protection of secondary amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectionSecondaryAmine(BaseScoring):
    """
    Evaluates synthesis routes for Boc protection of secondary amine functionality.
    Detects reactions where (Boc)2O is used to protect secondary amines with tert-butoxycarbonyl groups.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "relative")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to 0-10 score"""
        if x < 0:
            return 0  # Protection strategy not found
        
        if self.condition_type == "bool":
            return 1  # Protection found at any depth
        else:
            # Earlier protection (lower depth) is generally better
            return max(0, 1 - x)
    
    def hit_condition(self, d) -> bool:
        """Check if reaction involves Boc protection of secondary amine"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for (Boc)2O reagent pattern
            boc_anhydride_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)OC(=O)OC(C)(C)C")
            has_boc_reagent = any(mol.HasSubstructMatch(boc_anhydride_pattern) for mol in reactants)
            
            if not has_boc_reagent:
                return False
            
            # Check for secondary amine in reactants
            secondary_amine_pattern = Chem.MolFromSmarts("[CH1,CH2;!R][NH1][CH1,CH2;!R]")  # Acyclic secondary amine
            cyclic_secondary_amine_pattern = Chem.MolFromSmarts("[CH2][NH1][CH2]")  # Cyclic secondary amine
            
            has_secondary_amine = False
            for mol in reactants:
                if (mol.HasSubstructMatch(secondary_amine_pattern) or 
                    mol.HasSubstructMatch(cyclic_secondary_amine_pattern)):
                    # Make sure it's not the Boc reagent itself
                    if not mol.HasSubstructMatch(boc_anhydride_pattern):
                        has_secondary_amine = True
                        break
            
            if not has_secondary_amine:
                return False
            
            # Check for Boc-protected amine in products
            boc_protected_amine_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)N")
            has_boc_product = any(mol.HasSubstructMatch(boc_protected_amine_pattern) for mol in products)
            
            return has_boc_product
            
        except Exception:
            return False
