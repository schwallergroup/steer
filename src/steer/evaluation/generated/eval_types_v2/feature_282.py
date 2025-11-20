"""Generated evaluation code for: Alcohol as aldehyde protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AlcoholAldehydeProtection(BaseScoring):
    """
    Evaluates synthesis routes based on the use of alcohol as an aldehyde protecting group strategy.
    
    This class checks if benzyl alcohol or similar alcohol groups are used to mask aldehydes
    during reactions where aldehydes might interfere (e.g., under basic conditions).
    The strategy involves converting aldehydes to alcohols early in the synthesis and
    oxidizing them back to aldehydes at the appropriate stage.
    """
    
    def __init__(self, config: Dict):
        self.strategy = config["parameters"].get("strategy", "minimal")
        self.alcohol_pattern = Chem.MolFromSmarts("[CH2][OH]")  # Primary alcohol
        self.benzyl_alcohol_pattern = Chem.MolFromSmarts("c[CH2][OH]")  # Benzyl alcohol
        self.aldehyde_pattern = Chem.MolFromSmarts("[CH]=O")  # Aldehyde
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not found
        else:
            # Earlier implementation of protection strategy is better
            if self.strategy == "minimal":
                return max(0, 10 * (1 - x))  # Higher score for earlier protection
            else:
                return 5 * (1 - x)  # Moderate preference for earlier protection
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves alcohol-aldehyde protection strategy.
        
        Looks for:
        1. Reduction of aldehyde to alcohol (protection step)
        2. Oxidation of alcohol to aldehyde (deprotection step)
        3. Presence of alcohol where aldehyde would be expected
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".") if r.strip()]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".") if p.strip()]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for aldehyde to alcohol conversion (protection)
            if self._is_aldehyde_reduction(reactants, products):
                return True
                
            # Check for alcohol to aldehyde conversion (deprotection)
            if self._is_alcohol_oxidation(reactants, products):
                return True
                
            # Check if alcohol is being used in place where aldehyde might react
            if self._is_protected_aldehyde_reaction(reactants, products):
                return True
                
        except Exception:
            return False
            
        return False
    
    def _is_aldehyde_reduction(self, reactants, products) -> bool:
        """Check if reaction converts aldehyde to alcohol (protection step)."""
        reactant_aldehydes = sum(mol.GetSubstructMatches(self.aldehyde_pattern) 
                               for mol in reactants if mol)
        product_alcohols = sum(mol.GetSubstructMatches(self.alcohol_pattern) 
                             for mol in products if mol)
        
        # More alcohols in products than aldehydes lost from reactants suggests reduction
        return reactant_aldehydes > 0 and product_alcohols > reactant_aldehydes
    
    def _is_alcohol_oxidation(self, reactants, products) -> bool:
        """Check if reaction converts alcohol to aldehyde (deprotection step)."""
        reactant_alcohols = sum(mol.GetSubstructMatches(self.alcohol_pattern) 
                              for mol in reactants if mol)
        product_aldehydes = sum(mol.GetSubstructMatches(self.aldehyde_pattern) 
                              for mol in products if mol)
        
        # More aldehydes in products suggests oxidation of alcohols
        return reactant_alcohols > 0 and product_aldehydes > 0
    
    def _is_protected_aldehyde_reaction(self, reactants, products) -> bool:
        """Check if benzyl alcohol is participating in reactions where aldehyde would interfere."""
        # Look for benzyl alcohol in reactants (common aldehyde protecting group)
        has_benzyl_alcohol = any(mol.HasSubstructMatch(self.benzyl_alcohol_pattern) 
                               for mol in reactants if mol)
        
        # Check if this appears to be a reaction where aldehyde protection would be beneficial
        # (e.g., basic conditions, organometallic reactions)
        if has_benzyl_alcohol:
            # Look for patterns suggesting basic or nucleophilic conditions
            # This is a simplified check - in practice, you'd want more sophisticated analysis
            return True
            
        return False
