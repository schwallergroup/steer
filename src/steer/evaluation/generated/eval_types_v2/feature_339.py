"""Generated evaluation code for: Weinreb amide intermediate for selective ketone formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class WeinrebAmideFormation(BaseScoring):
    """
    Evaluates synthesis routes for the presence of Weinreb amide formation reactions.
    
    Detects the conversion of carboxylic acids (C(=O)O) to Weinreb amides (C(=O)N(C)OC)
    for strategic ketone formation. Returns depth-based scoring where earlier formation
    is preferred (lower depth = higher score).
    """
    
    def __init__(self, config: Dict):
        self.substrate_smarts = config["parameters"]["substrate_smarts"]  # "C(=O)O"
        self.product_smarts = config["parameters"]["product_smarts"]      # "C(=O)N(C)OC"
        
        # Compile SMARTS patterns for efficiency
        self.substrate_pattern = Chem.MolFromSmarts(self.substrate_smarts)
        self.product_pattern = Chem.MolFromSmarts(self.product_smarts)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        Earlier Weinreb amide formation is preferred for strategic synthesis.
        """
        if x < 0:
            return 0  # Weinreb amide formation not found
        else:
            # Earlier formation (lower depth) gets higher score
            # x=0 (at target) -> score=10, x=1 (at root) -> score=1
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents Weinreb amide formation.
        
        Args:
            d: Dictionary containing reaction metadata
            
        Returns:
            bool: True if this reaction forms a Weinreb amide from carboxylic acid
        """
        try:
            # Extract mapped reaction SMILES
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            # Split into products and reactants
            products_smiles, reactants_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".") if p.strip()]
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".") if r.strip()]
            
            # Filter out None molecules (parsing failures)
            products = [mol for mol in products if mol is not None]
            reactants = [mol for mol in reactants if mol is not None]
            
            if not products or not reactants:
                return False
            
            # Check if any product contains Weinreb amide pattern
            has_weinreb_product = any(mol.HasSubstructMatch(self.product_pattern) for mol in products)
            
            # Check if any reactant contains carboxylic acid pattern
            has_acid_reactant = any(mol.HasSubstructMatch(self.substrate_pattern) for mol in reactants)
            
            # Both conditions must be met for Weinreb amide formation
            return has_weinreb_product and has_acid_reactant
            
        except Exception:
            # Return False for any parsing or processing errors
            return False
