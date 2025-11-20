"""Generated evaluation code for: Late stage amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Evaluates synthesis routes for late-stage amide coupling reactions.
    
    This class checks if an amide bond formation occurs in the later stages
    of the synthesis route, with better scores for reactions happening closer
    to the final product.
    """
    
    def __init__(self, config: Dict):
        # For late-stage timing, we want the reaction to occur early in the tree
        # (which corresponds to late in the actual synthesis)
        self.target_depth = 0.2  # Within first 20% of route depth
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10 scale).
        
        Args:
            x: Depth fraction where amide coupling occurs (-1 if not found)
            
        Returns:
            Score from 0-10, with 10 being optimal late-stage timing
        """
        if x < 0:
            return 0  # No amide coupling found
        
        # For late-stage reactions, lower depth fractions are better
        if x <= self.target_depth:
            return 10  # Perfect late-stage timing
        else:
            # Linearly decrease score as reaction moves earlier in synthesis
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction node represents an amide coupling.
        
        Args:
            d: Reaction node dictionary containing metadata
            
        Returns:
            True if the reaction is an amide coupling
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
        
        try:
            # Split reaction SMILES
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            # Parse molecules
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
            
            return self._detect_amide_formation(prod_mol, react_mols)
            
        except Exception:
            return False
    
    def _detect_amide_formation(self, product, reactants):
        """
        Detect if amide bond formation occurred between reactants and product.
        
        Args:
            product: Product molecule (RDKit Mol object)
            reactants: List of reactant molecules (RDKit Mol objects)
            
        Returns:
            True if amide bond formation is detected
        """
        # Define amide bond pattern (C(=O)-N)
        amide_pattern = Chem.MolFromSmarts("[C](=[O])[N]")
        
        if not amide_pattern:
            return False
        
        # Check if product contains amide bonds
        product_amides = product.GetSubstructMatches(amide_pattern)
        if not product_amides:
            return False
        
        # Count amide bonds in reactants
        reactant_amide_count = 0
        for reactant in reactants:
            reactant_amide_count += len(reactant.GetSubstructMatches(amide_pattern))
        
        # Count amide bonds in product
        product_amide_count = len(product_amides)
        
        # Amide coupling should increase the number of amide bonds
        if product_amide_count > reactant_amide_count:
            # Additional check: look for carboxylic acid/ester + amine pattern
            return self._check_amide_coupling_precursors(reactants)
        
        return False
    
    def _check_amide_coupling_precursors(self, reactants):
        """
        Check if reactants contain typical amide coupling precursors.
        
        Args:
            reactants: List of reactant molecules
            
        Returns:
            True if carboxylic acid/ester and amine precursors are present
        """
        # Patterns for amide coupling precursors
        carboxylic_acid = Chem.MolFromSmarts("[C](=[O])[OH]")
        ester = Chem.MolFromSmarts("[C](=[O])[O][C]")
        acid_chloride = Chem.MolFromSmarts("[C](=[O])[Cl]")
        primary_amine = Chem.MolFromSmarts("[N]([H])[H]")
        secondary_amine = Chem.MolFromSmarts("[N]([H])([C])")
        
        has_acid_equivalent = False
        has_amine = False
        
        for reactant in reactants:
            # Check for acid equivalents
            if (reactant.HasSubstructMatch(carboxylic_acid) or 
                reactant.HasSubstructMatch(ester) or 
                reactant.HasSubstructMatch(acid_chloride)):
                has_acid_equivalent = True
            
            # Check for amines
            if (reactant.HasSubstructMatch(primary_amine) or 
                reactant.HasSubstructMatch(secondary_amine)):
                has_amine = True
        
        return has_acid_equivalent and has_amine
