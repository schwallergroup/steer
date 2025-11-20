"""Generated evaluation code for: Standard protecting group strategy for piperidine nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PiperidineProtectingGroupStrategy(BaseScoring):
    """
    Evaluates routes for standard protecting group strategy on piperidine nitrogen.
    Checks if carbamate protection of secondary amine occurs at appropriate depth,
    enabling selective functionalization before final alkylation.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config.get("protecting_group", "carbamate")
        self.functional_group = config.get("functional_group", "secondary_amine")
        self.timing = config.get("timing", "standard")
        
        # SMARTS patterns for detection
        self.piperidine_pattern = Chem.MolFromSmarts("[CH2]1[CH2][CH2][NH][CH2][CH2]1")  # Free piperidine NH
        self.carbamate_pattern = Chem.MolFromSmarts("[CH2]1[CH2][CH2]N([CH2][CH2]1)C(=O)O")  # Boc/Cbz protected piperidine
        self.alkyl_carbamate_pattern = Chem.MolFromSmarts("[CH2]1[CH2][CH2]N([CH2][CH2]1)C(=O)O[CH2,CH3]")  # Specific carbamate esters

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection doesn't occur
        
        if self.timing == "standard":
            # Standard timing: protection should occur in first half of synthesis
            if x <= 0.5:
                return 10 * (1 - 2 * x)  # Earlier is better, max 10 at depth 0
            else:
                return 2 * (1 - x)  # Late protection gets lower score
        else:
            # General case: moderate preference for earlier protection
            return 8 * (1 - x)

    def hit_condition(self, d) -> bool:
        """Check if this reaction involves carbamate protection of piperidine nitrogen"""
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            # Filter out None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Check for protection: free piperidine in reactants, protected in products
            has_free_piperidine_reactant = any(mol.HasSubstructMatch(self.piperidine_pattern) for mol in reactant_mols)
            has_protected_piperidine_product = any(mol.HasSubstructMatch(self.carbamate_pattern) or 
                                                 mol.HasSubstructMatch(self.alkyl_carbamate_pattern) for mol in product_mols)
            
            # Additional check for carbamate reagents (Boc2O, CbzCl, etc.)
            carbamate_reagent_patterns = [
                Chem.MolFromSmarts("CC(C)(C)OC(=O)OC(=O)OC(C)(C)C"),  # Boc2O
                Chem.MolFromSmarts("O=C(Cl)OCc1ccccc1"),  # CbzCl
                Chem.MolFromSmarts("CC(C)(C)OC(=O)Cl"),   # BocCl
            ]
            
            has_carbamate_reagent = any(
                any(mol.HasSubstructMatch(pattern) for mol in reactant_mols)
                for pattern in carbamate_reagent_patterns if pattern is not None
            )
            
            return has_free_piperidine_reactant and has_protected_piperidine_product and has_carbamate_reagent
            
        except Exception:
            return False
