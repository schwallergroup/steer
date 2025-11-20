"""Generated evaluation code for: Cbz protecting group strategy for amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CbzProtectingGroupStrategy(BaseScoring):
    """
    Evaluates synthesis routes for Cbz (benzyloxycarbonyl) protecting group strategy on primary amines.
    Checks if a primary amine is protected with Cbz early in the route and carried through multiple steps
    before deprotection.
    """
    
    def __init__(self, config: Dict):
        self.steps_protected = config["parameters"]["steps_protected"]
        self.cbz_pattern = Chem.MolFromSmarts("NC(=O)OCc1ccccc1")  # Cbz-protected amine
        self.primary_amine_pattern = Chem.MolFromSmarts("[NH2]")  # Primary amine
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not found
        # Score based on how close the protection depth is to target
        target_fraction = 1.0 - (self.steps_protected / 10.0)  # Convert steps to rough depth fraction
        return max(0, 10 - abs(x - target_fraction) * 10)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves Cbz protection of a primary amine
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol:
                    reactant_mols.append(mol)
                    
            product_mol = Chem.MolFromSmiles(products_smiles.strip())
            
            if not product_mol or not reactant_mols:
                return False
            
            # Check if product contains Cbz-protected amine
            has_cbz_product = product_mol.HasSubstructMatch(self.cbz_pattern)
            
            # Check if any reactant has primary amine
            has_primary_amine_reactant = any(
                mol.HasSubstructMatch(self.primary_amine_pattern) 
                for mol in reactant_mols
            )
            
            # Also check for presence of Cbz reagent (CbzCl or similar)
            cbz_reagent_pattern = Chem.MolFromSmarts("ClC(=O)OCc1ccccc1")  # CbzCl
            has_cbz_reagent = any(
                mol.HasSubstructMatch(cbz_reagent_pattern)
                for mol in reactant_mols
            )
            
            # This is a Cbz protection reaction if:
            # 1. Product has Cbz-protected amine
            # 2. Reactant has primary amine
            # 3. Cbz reagent is present
            return (has_cbz_product and 
                   has_primary_amine_reactant and 
                   has_cbz_reagent)
                   
        except Exception:
            return False
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Override to check if the Cbz protection occurs early enough and
        the protected group persists through the required number of steps
        """
        def bfs_protection_strategy(node, depth=0):
            # Check if current node is Cbz protection
            if self.hit_condition(node):
                return self._check_protection_persistence(node, depth)
            
            # Continue BFS through children
            children = node.get("children", [])
            for child in children:
                result = bfs_protection_strategy(child, depth + 1)
                if result[0]:  # Found valid strategy
                    return result
                    
            return False, -1
        
        return bfs_protection_strategy(d)
    
    def _check_protection_persistence(self, protection_node, protection_depth) -> Tuple[bool, int]:
        """
        Check if the Cbz group persists through the required number of steps
        """
        def count_protected_steps(node, steps_count=0):
            # Check if current reaction removes Cbz protection
            if self._is_cbz_deprotection(node):
                return steps_count >= self.steps_protected, steps_count
            
            # Continue through children while Cbz is still present
            children = node.get("children", [])
            for child in children:
                if self._molecule_has_cbz_protection(child):
                    return count_protected_steps(child, steps_count + 1)
            
            return False, steps_count
        
        # Start counting from the protection step
        is_valid, steps = count_protected_steps(protection_node)
        depth_fraction = protection_depth / 10.0  # Convert to fraction
        
        return is_valid, depth_fraction
    
    def _is_cbz_deprotection(self, node) -> bool:
        """Check if this reaction removes Cbz protection"""
        metadata = node.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            reactant_mol = Chem.MolFromSmiles(reactants_smiles.split(".")[0])
            product_mol = Chem.MolFromSmiles(products_smiles)
            
            if not reactant_mol or not product_mol:
                return False
            
            # Cbz deprotection: reactant has Cbz, product has free amine
            has_cbz_reactant = reactant_mol.HasSubstructMatch(self.cbz_pattern)
            has_free_amine_product = product_mol.HasSubstructMatch(self.primary_amine_pattern)
            
            return has_cbz_reactant and has_free_amine_product
            
        except Exception:
            return False
    
    def _molecule_has_cbz_protection(self, node) -> bool:
        """Check if the main molecule in this node has Cbz protection"""
        try:
            # Get the main product molecule
            metadata = node.get("metadata", {})
            mapped_rxn = metadata.get("mapped_reaction_smiles", "")
            
            if mapped_rxn and ">>" in mapped_rxn:
                products_smiles = mapped_rxn.split(">>")[1]
                product_mol = Chem.MolFromSmiles(products_smiles)
                
                if product_mol:
                    return product_mol.HasSubstructMatch(self.cbz_pattern)
            
            return False
        except Exception:
            return False
