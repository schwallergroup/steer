"""Generated evaluation code for: Boc protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectingGroupStrategy(BaseScoring):
    """
    Evaluates synthesis routes for proper use of Boc protecting group strategy.
    Checks if Boc protection of secondary amine occurs before interference-prone reactions
    like Williamson ether synthesis.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.functional_group = config["parameters"]["functional_group"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not implemented
        else:
            return 1 - x  # Earlier protection is better (higher score)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves Boc protection of secondary amine"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(mol.strip()) for mol in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(mol.strip()) for mol in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Check for Boc reagent in reactants (tert-butyl dicarbonate or Boc-Cl)
            boc_reagent_patterns = [
                "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C",  # Boc2O
                "CC(C)(C)OC(=O)Cl"  # Boc-Cl
            ]
            
            has_boc_reagent = False
            for reactant in reactant_mols:
                for pattern in boc_reagent_patterns:
                    boc_mol = Chem.MolFromSmiles(pattern)
                    if boc_mol and reactant.HasSubstructMatch(boc_mol):
                        has_boc_reagent = True
                        break
                if has_boc_reagent:
                    break
            
            if not has_boc_reagent:
                return False
            
            # Check for secondary amine in reactants and Boc-protected amine in products
            secondary_amine_pattern = Chem.MolFromSmarts("[C,c][NH][C,c]")  # Secondary amine
            boc_protected_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)N([C,c])[C,c]")  # Boc-protected amine
            
            has_secondary_amine = False
            for reactant in reactant_mols:
                if reactant.HasSubstructMatch(secondary_amine_pattern):
                    has_secondary_amine = True
                    break
            
            has_boc_product = False
            for product in product_mols:
                if product.HasSubstructMatch(boc_protected_pattern):
                    has_boc_product = True
                    break
            
            return has_boc_reagent and has_secondary_amine and has_boc_product
            
        except Exception:
            return False
