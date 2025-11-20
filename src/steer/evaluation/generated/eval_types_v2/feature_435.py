"""Generated evaluation code for: Boc protection for chemoselectivity control"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectionStrategy(BaseScoring):
    """
    Evaluates synthesis routes for Boc protection strategy on amines.
    Checks if Boc protection is used before chemically sensitive reactions
    and deprotected at an appropriate stage.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.3)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Boc protection strategy not used
        else:
            # Earlier use of Boc protection is generally better for chemoselectivity
            if self.condition_type == "bool":
                return 1  # Strategy is present
            else:
                return max(0, 1 - x)  # Earlier depth gets higher score
    
    def hit_condition(self, d):
        """Check if this reaction involves Boc protection of an amine"""
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
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) 
                           for smi in reactants_smiles.split(".") if smi.strip()]
            product_mols = [Chem.MolFromSmiles(smi.strip()) 
                          for smi in products_smiles.split(".") if smi.strip()]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Boc group SMARTS pattern: tert-butoxycarbonyl
            boc_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)N")
            # Free amine pattern
            amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
            
            if not boc_pattern or not amine_pattern:
                return False
            
            # Check for Boc protection: amine in reactants, Boc-protected amine in products
            reactant_has_free_amine = any(mol.HasSubstructMatch(amine_pattern) 
                                        for mol in reactant_mols)
            product_has_boc = any(mol.HasSubstructMatch(boc_pattern) 
                                for mol in product_mols)
            
            # Also check for Boc reagent in reactants (Boc2O or Boc-Cl)
            boc_reagent_patterns = [
                Chem.MolFromSmarts("CC(C)(C)OC(=O)OC(=O)OC(C)(C)C"),  # Boc2O
                Chem.MolFromSmarts("CC(C)(C)OC(=O)Cl")  # Boc-Cl
            ]
            
            reactant_has_boc_reagent = any(
                mol.HasSubstructMatch(pattern) 
                for mol in reactant_mols 
                for pattern in boc_reagent_patterns 
                if pattern
            )
            
            return (reactant_has_free_amine and 
                   product_has_boc and 
                   reactant_has_boc_reagent)
            
        except Exception:
            return False
