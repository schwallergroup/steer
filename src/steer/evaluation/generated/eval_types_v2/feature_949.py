"""Generated evaluation code for: Early imidazole protection strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyImidazoleProtection(BaseScoring):
    """
    Evaluates synthesis routes for early protection of imidazole NH with Boc group.
    Checks if Boc protection of imidazole occurs in the early stages of synthesis
    to enable selective transformations while preventing unwanted N-alkylation.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config.get("protecting_group", "Boc")
        self.timing_threshold = config.get("timing_threshold", 0.3)  # Early = first 30% of route
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection doesn't happen
        else:
            # Earlier protection gets higher score
            # x is depth fraction, so smaller x = earlier = better
            if x <= self.timing_threshold:
                return 10 * (1 - x / self.timing_threshold)  # Scale 10-0 for early timing
            else:
                return 0  # Too late for early protection strategy
    
    def hit_condition(self, d):
        """Check if this reaction involves Boc protection of imidazole NH"""
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
            
            if None in reactant_mols or None in product_mols:
                return False
            
            # Check for imidazole in reactants and Boc-protected imidazole in products
            imidazole_pattern = Chem.MolFromSmarts("[nH]1ccnc1")  # Imidazole NH pattern
            boc_imidazole_pattern = Chem.MolFromSmarts("n1ccnc1C(=O)OC(C)(C)C")  # Boc-protected imidazole
            
            # Alternative Boc pattern (more general)
            boc_pattern = Chem.MolFromSmarts("C(=O)OC(C)(C)C")  # Boc group
            
            has_imidazole_reactant = any(mol.HasSubstructMatch(imidazole_pattern) for mol in reactant_mols)
            
            # Check for Boc group formation in products
            has_boc_product = any(mol.HasSubstructMatch(boc_pattern) for mol in product_mols)
            
            # Additional check: ensure imidazole nitrogen is protected
            protected_imidazole = False
            for prod_mol in product_mols:
                if prod_mol.HasSubstructMatch(boc_pattern):
                    # Check if the same molecule also has imidazole-like structure
                    imidazole_carbon_pattern = Chem.MolFromSmarts("n1ccnc1")  # Imidazole without NH
                    if prod_mol.HasSubstructMatch(imidazole_carbon_pattern):
                        protected_imidazole = True
                        break
            
            return has_imidazole_reactant and has_boc_product and protected_imidazole
            
        except Exception:
            return False
