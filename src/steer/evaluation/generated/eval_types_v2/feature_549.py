"""Generated evaluation code for: Weinreb amide to aldehyde selective reduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class WeinrebAmideReduction(BaseScoring):
    """
    Evaluates synthesis routes for Weinreb amide to aldehyde selective reduction.
    
    This scoring function identifies reactions that use Weinreb amide intermediates
    to achieve selective reduction to aldehydes in the presence of ketone functionality.
    The Weinreb amide (N-methoxy-N-methylamide) allows for controlled reduction
    that stops at the aldehyde stage without over-reduction.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.0)
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to 0-10 score, favoring earlier use of Weinreb reduction."""
        if x < 0:
            return 0  # Reaction not found
        else:
            # Earlier use of selective reduction is better (closer to target molecule)
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction involves Weinreb amide reduction to aldehyde.
        
        Looks for:
        1. Weinreb amide (N-methoxy-N-methylamide) in reactants
        2. Corresponding aldehyde in products
        3. Preservation of ketone groups (selectivity check)
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Define Weinreb amide pattern: N-methoxy-N-methylamide
            weinreb_pattern = Chem.MolFromSmarts("[C;X3]=[O;X1]-[N;X3](-[CH3])-[O;X2]-[CH3]")
            if weinreb_pattern is None:
                return False
            
            # Define aldehyde pattern
            aldehyde_pattern = Chem.MolFromSmarts("[C;X3]=[O;X1]-[H]")
            if aldehyde_pattern is None:
                return False
                
            # Define ketone pattern for selectivity check
            ketone_pattern = Chem.MolFromSmarts("[C;X4]-[C;X3]=[O;X1]")
            if ketone_pattern is None:
                return False
            
            # Check for Weinreb amide in reactants
            has_weinreb_reactant = any(mol.HasSubstructMatch(weinreb_pattern) for mol in reactants)
            if not has_weinreb_reactant:
                return False
            
            # Check for aldehyde in products
            has_aldehyde_product = any(mol.HasSubstructMatch(aldehyde_pattern) for mol in products)
            if not has_aldehyde_product:
                return False
            
            # Selectivity check: if ketones are present in reactants, they should be preserved in products
            reactant_ketone_count = sum(len(mol.GetSubstructMatches(ketone_pattern)) for mol in reactants)
            product_ketone_count = sum(len(mol.GetSubstructMatches(ketone_pattern)) for mol in products)
            
            # If there were ketones in reactants, they should be preserved (demonstrating selectivity)
            if reactant_ketone_count > 0 and product_ketone_count < reactant_ketone_count:
                return False  # Ketones were also reduced, not selective
            
            return True
            
        except Exception:
            return False
