"""Generated evaluation code for: Protecting group swap from Cbz to Boc"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupSwap(BaseScoring):
    """
    Detects protecting group swap from Cbz to Boc on amine functional groups.
    Checks if the route contains both deprotection of Cbz and protection with Boc
    within a reasonable sequence of reactions.
    """
    
    def __init__(self, config: Dict):
        self.initial_group = config["initial_group"]  # "Cbz"
        self.final_group = config["final_group"]      # "Boc"
        self.functional_group = config["functional_group"]  # "amine"
        self.swap_occurs = config["swap_occurs"]      # True/False
        
        # Define SMARTS patterns for protecting groups
        self.cbz_pattern = Chem.MolFromSmarts("C(=O)OCc1ccccc1")  # Cbz pattern
        self.boc_pattern = Chem.MolFromSmarts("C(=O)OC(C)(C)C")   # Boc pattern
        self.amine_pattern = Chem.MolFromSmarts("N")               # Amine nitrogen
    
    def route_scoring(self, x) -> float:
        """
        Convert swap detection result to score.
        x: 1 if swap detected, 0 if not detected, -1 if condition not met
        """
        if x < 0:
            return 0 if self.swap_occurs else 10  # No swap found
        elif x == 1:
            return 10 if self.swap_occurs else 0  # Swap detected
        else:
            return 0 if self.swap_occurs else 10  # No swap detected
    
    def hit_condition(self, d) -> bool:
        """
        Check if protecting group swap occurs by analyzing the entire route tree.
        Returns True if swap is detected.
        """
        # Get all reactions in the route
        all_reactions = self._get_all_reactions(d)
        
        cbz_deprotection_found = False
        boc_protection_found = False
        
        for rxn_data in all_reactions:
            if self._is_cbz_deprotection(rxn_data):
                cbz_deprotection_found = True
            elif self._is_boc_protection(rxn_data):
                boc_protection_found = True
        
        # Swap occurs if both deprotection and protection are found
        return cbz_deprotection_found and boc_protection_found
    
    def _get_all_reactions(self, node):
        """Recursively collect all reaction data from the route tree."""
        reactions = []
        
        if "metadata" in node and "mapped_reaction_smiles" in node["metadata"]:
            reactions.append(node)
        
        # Traverse children
        if "children" in node:
            for child in node["children"]:
                reactions.extend(self._get_all_reactions(child))
        
        return reactions
    
    def _is_cbz_deprotection(self, rxn_data) -> bool:
        """Check if reaction involves Cbz deprotection."""
        try:
            rxn_smiles = rxn_data["metadata"]["mapped_reaction_smiles"]
            reactants, products = rxn_smiles.split(">>")
            
            reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi) for smi in products.split(".")]
            
            # Check if Cbz is present in reactants but not in main product
            cbz_in_reactants = any(mol and mol.HasSubstructMatch(self.cbz_pattern) 
                                 for mol in reactant_mols if mol)
            
            # Check if free amine is formed (simplified check)
            if cbz_in_reactants:
                # Look for loss of Cbz group - this is a simplified heuristic
                return True
            
            return False
            
        except Exception:
            return False
    
    def _is_boc_protection(self, rxn_data) -> bool:
        """Check if reaction involves Boc protection."""
        try:
            rxn_smiles = rxn_data["metadata"]["mapped_reaction_smiles"]
            reactants, products = rxn_smiles.split(">>")
            
            reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi) for smi in products.split(".")]
            
            # Check if Boc group appears in products but not in main reactant
            boc_in_products = any(mol and mol.HasSubstructMatch(self.boc_pattern) 
                                for mol in product_mols if mol)
            
            # Check for Boc reagent in reactants (like Boc2O)
            boc_reagent_patterns = [
                Chem.MolFromSmarts("C(=O)OC(=O)OC(C)(C)C"),  # Boc2O
                Chem.MolFromSmarts("C(=O)OC(C)(C)C")          # Boc-containing reagent
            ]
            
            boc_reagent_present = any(
                any(mol and mol.HasSubstructMatch(pattern) for mol in reactant_mols if mol)
                for pattern in boc_reagent_patterns
            )
            
            return boc_in_products and boc_reagent_present
            
        except Exception:
            return False
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Override to check for swap occurrence across the entire route.
        Returns (swap_detected, route_length).
        """
        swap_detected = self.hit_condition(d)
        route_length = self._calculate_route_depth(d)
        
        return swap_detected, route_length
    
    def _calculate_route_depth(self, node, depth=0) -> int:
        """Calculate the maximum depth of the route tree."""
        if "children" not in node or not node["children"]:
            return depth
        
        return max(self._calculate_route_depth(child, depth + 1) 
                  for child in node["children"])
