"""Generated evaluation code for: Chiral auxiliary diastereoselective alkylation strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ChiralAuxiliaryAlkylation(BaseScoring):
    """
    Evaluates routes for chiral auxiliary diastereoselective alkylation strategy.
    Checks for alkylation reactions involving phenylethyl chiral auxiliary and allyl bromide.
    """
    
    def __init__(self, config: Dict):
        self.reaction_type = config["parameters"]["reaction_type"]
        self.chiral_auxiliary = config["parameters"]["chiral_auxiliary"]
        self.alkylating_agent = config["parameters"]["alkylating_agent"]
        
        # Define SMARTS patterns for detection
        self.phenylethyl_pattern = Chem.MolFromSmarts("c1ccccc1[CH](C)[NH]")  # Phenylethyl auxiliary
        self.allyl_bromide_pattern = Chem.MolFromSmarts("C=CCBr")  # Allyl bromide
        self.alkylation_product_pattern = Chem.MolFromSmarts("C=CC[CH]")  # Allyl group attached to carbon
    
    def route_scoring(self, x) -> float:
        """
        Score based on depth of chiral auxiliary alkylation.
        Earlier occurrence (lower depth) scores higher.
        """
        if x < 0:
            return 0  # Reaction not found
        else:
            return 1 - x  # Earlier alkylation is better for stereoselectivity
    
    def hit_condition(self, d) -> bool:
        """
        Check if reaction represents chiral auxiliary diastereoselective alkylation.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in react_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for phenylethyl auxiliary in product
            has_chiral_auxiliary = product.HasSubstructMatch(self.phenylethyl_pattern)
            
            # Check for allyl bromide in reactants
            has_allyl_bromide = any(r.HasSubstructMatch(self.allyl_bromide_pattern) for r in reactants)
            
            # Check for allyl group incorporation in product
            has_allyl_product = product.HasSubstructMatch(self.alkylation_product_pattern)
            
            # Check for alkylation pattern: C-C bond formation
            # Look for carbon with increased substitution in product vs reactants
            alkylation_occurred = self._detect_alkylation_pattern(product, reactants)
            
            return (has_chiral_auxiliary and 
                   has_allyl_bromide and 
                   has_allyl_product and 
                   alkylation_occurred)
                   
        except Exception:
            return False
    
    def _detect_alkylation_pattern(self, product, reactants) -> bool:
        """
        Detect if alkylation has occurred by comparing carbon substitution patterns.
        """
        try:
            # Count carbons with specific substitution patterns
            prod_carbons = self._count_substituted_carbons(product)
            
            total_react_carbons = 0
            for reactant in reactants:
                if reactant.GetNumAtoms() > 1:  # Skip simple molecules like Br-
                    total_react_carbons += self._count_substituted_carbons(reactant)
            
            # Alkylation typically increases the number of substituted carbons
            return prod_carbons >= total_react_carbons
            
        except Exception:
            return False
    
    def _count_substituted_carbons(self, mol) -> int:
        """
        Count carbons with 3 or 4 non-hydrogen substituents (quaternary/tertiary).
        """
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                degree = atom.GetDegree()
                if degree >= 3:  # Tertiary or quaternary carbon
                    count += 1
        return count
