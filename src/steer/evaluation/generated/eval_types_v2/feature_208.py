"""Generated evaluation code for: Strategic Boc protection for selectivity control"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectionStrategy(BaseScoring):
    """
    Evaluates strategic Boc protection of secondary amines for selectivity control.
    Checks if Boc protection occurs at an appropriate depth in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "numeric")
        self.target_depth = config.get("target_depth", {}).get("value", 0.3)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No Boc protection found
        
        if self.condition_type == "bool":
            return 1  # Protection found regardless of depth
        else:
            # Earlier protection (lower depth) is generally better for selectivity
            return max(0, 1 - x)
    
    def hit_condition(self, d):
        """Check if this reaction involves Boc protection of a secondary amine."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    reactant_mols.append(mol)
            
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    product_mols.append(mol)
            
            # Check for Boc protection reaction
            return self._is_boc_protection_reaction(reactant_mols, product_mols)
            
        except Exception:
            return False
    
    def _is_boc_protection_reaction(self, reactants, products):
        """Check if reaction involves Boc protection of secondary amine."""
        # Secondary amine pattern
        secondary_amine_pattern = Chem.MolFromSmarts("[C,c][NH][C,c]")
        
        # Boc-protected secondary amine pattern
        boc_protected_pattern = Chem.MolFromSmarts("[C,c][N]([C](=O)[O][C]([C])([C])[C])[C,c]")
        
        # Boc reagent patterns (common Boc protection reagents)
        boc_reagent_patterns = [
            Chem.MolFromSmarts("CC(C)(C)OC(=O)OC(=O)OC(C)(C)C"),  # Boc2O
            Chem.MolFromSmarts("CC(C)(C)OC(=O)Cl"),  # Boc-Cl
        ]
        
        if not secondary_amine_pattern or not boc_protected_pattern:
            return False
        
        # Check if reactants contain secondary amine and Boc reagent
        has_secondary_amine = False
        has_boc_reagent = False
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(secondary_amine_pattern):
                has_secondary_amine = True
            
            for boc_pattern in boc_reagent_patterns:
                if boc_pattern and reactant.HasSubstructMatch(boc_pattern):
                    has_boc_reagent = True
        
        # Check if products contain Boc-protected amine
        has_boc_protected = False
        for product in products:
            if product.HasSubstructMatch(boc_protected_pattern):
                has_boc_protected = True
        
        # Must have secondary amine + Boc reagent -> Boc-protected product
        return has_secondary_amine and has_boc_reagent and has_boc_protected
