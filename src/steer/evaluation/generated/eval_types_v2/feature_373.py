"""Generated evaluation code for: Final protecting group addition step"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectionFinalStep(BaseScoring):
    """
    Evaluates if Boc protection is performed as the final synthetic step.
    Returns higher scores when Boc protection occurs later in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config.get("protecting_group", "Boc")
        self.timing = config.get("timing", "final")
        self.direction = config.get("direction", "addition")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Boc protection doesn't happen
        else:
            # Later stage protection is better (closer to 1.0 depth fraction)
            # Scale to 0-10 range with higher scores for later timing
            return 10 * x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves Boc protection addition"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1].split(".")
            
            # Check for Boc addition pattern
            return self._is_boc_protection_reaction(reactants, products)
            
        except (KeyError, AttributeError):
            return False
    
    def _is_boc_protection_reaction(self, reactants: str, products: List[str]) -> bool:
        """Detect if reaction involves Boc group addition to amine"""
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Boc reagent patterns (Boc2O, Boc-Cl, etc.)
            boc_reagents = [
                "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C",  # Boc2O
                "CC(C)(C)OC(=O)Cl",  # Boc-Cl
                "CC(C)(C)OC(=O)N1C(=O)CCC1=O"  # Boc-OSu
            ]
            
            # Check if any reactant is a Boc reagent
            has_boc_reagent = False
            for reactant in reactant_mols:
                for boc_pattern in boc_reagents:
                    boc_mol = Chem.MolFromSmiles(boc_pattern)
                    if boc_mol and reactant.HasSubstructMatch(boc_mol):
                        has_boc_reagent = True
                        break
            
            if not has_boc_reagent:
                return False
            
            # Check if products contain Boc-protected amine
            boc_carbamate_pattern = Chem.MolFromSmarts("NC(=O)OC(C)(C)C")
            
            # Count Boc groups in reactants vs products
            reactant_boc_count = sum(
                len(mol.GetSubstructMatches(boc_carbamate_pattern)) 
                for mol in reactant_mols if mol
            )
            
            product_boc_count = sum(
                len(mol.GetSubstructMatches(boc_carbamate_pattern))
                for mol in product_mols if mol
            )
            
            # Boc addition should increase the count
            return product_boc_count > reactant_boc_count
            
        except Exception:
            return False
