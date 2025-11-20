"""Generated evaluation code for: Boc protecting group strategy employed"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectingGroupStrategy(BaseScoring):
    """
    Evaluates synthesis routes for the use of Boc (tert-butoxycarbonyl) protecting group strategy.
    
    This scorer identifies reactions where:
    1. Boc protection is applied to an amine group
    2. Boc deprotection occurs to reveal the amine
    
    The strategy is considered successful when both protection and deprotection steps are present.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.functional_group = config["parameters"]["functional_group"]
        
        # SMARTS patterns for Boc group and related transformations
        self.boc_pattern = Chem.MolFromSmarts("[NH1,NH2][C](=O)OC(C)(C)C")  # Boc-protected amine
        self.free_amine_pattern = Chem.MolFromSmarts("[NH1,NH2]")  # Free amine
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score.
        Earlier use of Boc strategy (lower depth) gets higher score.
        """
        if x < 0:
            return 0  # Strategy not found
        else:
            return 1 - x  # Earlier strategy application is better
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves Boc protecting group strategy.
        Returns True if either Boc protection or deprotection is detected.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants_smiles, products_smiles = mapped_rxn.split(">>")
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            if not reactant_mols or not product_mols:
                return False
                
            # Check for Boc protection: free amine → Boc-protected amine
            boc_protection = self._detect_boc_protection(reactant_mols, product_mols)
            
            # Check for Boc deprotection: Boc-protected amine → free amine
            boc_deprotection = self._detect_boc_deprotection(reactant_mols, product_mols)
            
            return boc_protection or boc_deprotection
            
        except Exception:
            return False
    
    def _detect_boc_protection(self, reactants, products) -> bool:
        """
        Detect Boc protection: transformation from free amine to Boc-protected amine.
        """
        # Check if reactants have free amine and products have Boc-protected amine
        has_free_amine_reactant = any(mol.HasSubstructMatch(self.free_amine_pattern) for mol in reactants)
        has_boc_product = any(mol.HasSubstructMatch(self.boc_pattern) for mol in products)
        
        # Additional check: Boc reagent (tert-butyl dicarbonate) in reactants
        boc_reagent_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)OC(=O)OC(C)(C)C")  # Boc2O
        has_boc_reagent = any(mol.HasSubstructMatch(boc_reagent_pattern) for mol in reactants)
        
        return has_free_amine_reactant and has_boc_product and has_boc_reagent
    
    def _detect_boc_deprotection(self, reactants, products) -> bool:
        """
        Detect Boc deprotection: transformation from Boc-protected amine to free amine.
        """
        # Check if reactants have Boc-protected amine and products have free amine
        has_boc_reactant = any(mol.HasSubstructMatch(self.boc_pattern) for mol in reactants)
        has_free_amine_product = any(mol.HasSubstructMatch(self.free_amine_pattern) for mol in products)
        
        # Additional check: acid conditions (common deprotection reagents)
        acid_patterns = [
            Chem.MolFromSmarts("Cl"),  # HCl
            Chem.MolFromSmarts("F[C](F)(F)C(=O)O"),  # TFA
        ]
        has_acid = any(any(mol.HasSubstructMatch(pattern) for pattern in acid_patterns) for mol in reactants)
        
        return has_boc_reactant and has_free_amine_product and has_acid
