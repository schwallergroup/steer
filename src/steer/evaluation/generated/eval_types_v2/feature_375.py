"""Generated evaluation code for: Late stage protecting group addition"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageProtectingGroupAddition(BaseScoring):
    """
    Evaluates synthesis routes for late-stage protecting group addition.
    Specifically looks for carbamate (e.g., Boc) protection as a final step,
    which is unusual since protecting groups are typically added early and removed late.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group_type = config.get("protecting_group_type", "carbamate")
        self.timing = config.get("timing", "final_step")
        self.operation = config.get("operation", "addition")
        
        # Define SMARTS patterns for different protecting groups
        self.pg_patterns = {
            "carbamate": "[NH1,NH2][C](=O)O[CH3,C(CH3)3,CH2Ph]",  # Boc, Cbz, methyl carbamate
            "acetyl": "[NH1][C](=O)[CH3]",  # Acetyl protection
            "silyl": "[OH,NH][Si]([CH3])([CH3])[C(CH3)3]",  # TBS, TBDMS
            "benzyl": "[OH,NH][CH2]c1ccccc1"  # Benzyl protection
        }
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10). Earlier protection gets higher penalty."""
        if x < 0:
            return 0  # Protection doesn't happen
        else:
            # Late-stage protection (higher x) gets better score
            # x close to 1.0 (final step) should give high score
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves protecting group addition."""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants_smiles = rxn[0]
            products_smiles = rxn[1]
            
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p) for p in products_smiles.split(".")]
            
            # Remove None values from failed parsing
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check if protecting group is present in products but not in main reactant
            pg_pattern = Chem.MolFromSmarts(self.pg_patterns[self.protecting_group_type])
            if pg_pattern is None:
                return False
            
            # Find the main organic molecule (largest by atom count)
            main_reactant = max(reactants, key=lambda mol: mol.GetNumAtoms())
            main_product = max(products, key=lambda mol: mol.GetNumAtoms())
            
            # Check if protecting group was added
            reactant_has_pg = main_reactant.HasSubstructMatch(pg_pattern)
            product_has_pg = main_product.HasSubstructMatch(pg_pattern)
            
            # Protection addition: PG absent in reactant, present in product
            if self.operation == "addition":
                return not reactant_has_pg and product_has_pg
            # Protection removal: PG present in reactant, absent in product  
            elif self.operation == "removal":
                return reactant_has_pg and not product_has_pg
            
            return False
            
        except (KeyError, AttributeError, IndexError):
            return False
