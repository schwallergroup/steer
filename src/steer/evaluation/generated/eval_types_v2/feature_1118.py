"""Generated evaluation code for: TBS protecting group for phenol throughout synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TBSPhenolProtection(BaseScoring):
    """
    Evaluates TBS protecting group strategy for phenols in synthesis routes.
    Checks if TBS protection occurs at the specified depth and deprotection 
    occurs at the target deprotection depth.
    """
    
    def __init__(self, config: Dict):
        self.protection_depth = config["parameters"]["protection_depth"]
        self.deprotection_depth = config["parameters"]["deprotection_depth"]
        self.phenol_pattern = Chem.MolFromSmarts("[OH1][c]")  # Phenol OH
        self.tbs_pattern = Chem.MolFromSmarts("[Si](C)(C)C(C)(C)C")  # TBS group
        self.tbs_phenol_pattern = Chem.MolFromSmarts("[O][Si](C)(C)C(C)(C)C")  # TBS-protected oxygen
        
    def route_scoring(self, x) -> float:
        # x contains tuple of (protection_found, deprotection_found, protection_depth, deprotection_depth)
        if isinstance(x, tuple):
            protection_found, deprotection_found, prot_depth, deprot_depth = x
            
            if not protection_found:
                return 0  # No TBS protection found
                
            if not deprotection_found:
                return 5  # Partial score if protection but no deprotection
                
            # Score based on how close depths are to targets
            prot_score = max(0, 5 - abs(prot_depth - self.protection_depth))
            deprot_score = max(0, 5 - abs(deprot_depth - self.deprotection_depth))
            
            return (prot_score + deprot_score) / 2
        
        return 0
    
    def condition_depth(self, d) -> Tuple[bool, Any]:
        """
        Traverse the entire route to find TBS protection and deprotection events.
        """
        protection_found = False
        deprotection_found = False
        protection_depth = -1
        deprotection_depth = -1
        
        def traverse_route(node, depth=0):
            nonlocal protection_found, deprotection_found, protection_depth, deprotection_depth
            
            if "children" in node and node["children"]:
                for child in node["children"]:
                    if "metadata" in child and "mapped_reaction_smiles" in child["metadata"]:
                        if self.is_tbs_protection(child):
                            protection_found = True
                            protection_depth = depth
                        elif self.is_tbs_deprotection(child):
                            deprotection_found = True
                            deprotection_depth = depth
                    
                    traverse_route(child, depth + 1)
        
        traverse_route(d)
        
        condition_met = protection_found or deprotection_found
        result = (protection_found, deprotection_found, protection_depth, deprotection_depth)
        
        return condition_met, result
    
    def is_tbs_protection(self, reaction_node) -> bool:
        """
        Check if reaction involves TBS protection of phenol.
        Protection: phenol OH -> TBS-protected oxygen
        """
        try:
            rxn_smiles = reaction_node["metadata"]["mapped_reaction_smiles"]
            reactants, products = rxn_smiles.split(">>")
            
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Check if reactants have phenol and products have TBS-protected oxygen
            has_phenol_reactant = any(mol and mol.HasSubstructMatch(self.phenol_pattern) 
                                    for mol in reactant_mols if mol)
            has_tbs_product = any(mol and mol.HasSubstructMatch(self.tbs_phenol_pattern) 
                                for mol in product_mols if mol)
            has_tbs_reagent = any(mol and mol.HasSubstructMatch(self.tbs_pattern) 
                                for mol in reactant_mols if mol)
            
            return has_phenol_reactant and has_tbs_product and has_tbs_reagent
            
        except Exception:
            return False
    
    def is_tbs_deprotection(self, reaction_node) -> bool:
        """
        Check if reaction involves TBS deprotection to reveal phenol.
        Deprotection: TBS-protected oxygen -> phenol OH
        """
        try:
            rxn_smiles = reaction_node["metadata"]["mapped_reaction_smiles"]
            reactants, products = rxn_smiles.split(">>")
            
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Check if reactants have TBS-protected oxygen and products have phenol
            has_tbs_reactant = any(mol and mol.HasSubstructMatch(self.tbs_phenol_pattern) 
                                 for mol in reactant_mols if mol)
            has_phenol_product = any(mol and mol.HasSubstructMatch(self.phenol_pattern) 
                                   for mol in product_mols if mol)
            
            return has_tbs_reactant and has_phenol_product
            
        except Exception:
            return False
