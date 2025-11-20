"""Generated evaluation code for: Late stage amide coupling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Evaluates whether amide coupling occurs as the final step in synthesis.
    Detects formation of amide bonds (C-N bond formation with C=O present)
    and scores based on when this transformation occurs in the route.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "final_step")
        self.position = config.get("position", None)  # Optional specific position like C7
        
    def route_scoring(self, x) -> float:
        """
        Score based on timing of amide coupling.
        For final step timing, later is better (higher score).
        """
        if x < 0:
            return 0  # Amide coupling doesn't happen
        
        if self.timing == "final_step":
            # Final step gets highest score, earlier steps get lower scores
            return 1 - x  # x is depth fraction, so 1-x rewards later steps
        else:
            # For other timing requirements, can be extended
            return 1 - x
            
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents an amide coupling.
        Looks for formation of amide bond (C-N with adjacent C=O).
        """
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Check if amide bond is formed
            return self._is_amide_coupling(reactants, product)
            
        except Exception:
            return False
            
    def _is_amide_coupling(self, reactants, product) -> bool:
        """
        Detect if reaction forms an amide bond by comparing reactants to product.
        """
        # Look for amide pattern in product
        amide_pattern = Chem.MolFromSmarts("[C](=[O])[N]")
        if not product.HasSubstructMatch(amide_pattern):
            return False
            
        # Check that amide bond is newly formed (not present in reactants)
        amide_bonds_product = self._count_amide_bonds(product)
        amide_bonds_reactants = sum(self._count_amide_bonds(r) for r in reactants)
        
        # Must have net increase in amide bonds
        if amide_bonds_product <= amide_bonds_reactants:
            return False
            
        # Additional check: look for typical amide coupling patterns
        # Carboxylic acid/ester + amine -> amide + leaving group
        has_carbonyl_reactant = any(r.HasSubstructMatch(Chem.MolFromSmarts("[C](=[O])[OH,OR]")) for r in reactants)
        has_amine_reactant = any(r.HasSubstructMatch(Chem.MolFromSmarts("[N;!$(N=*);!$(N#*)]")) for r in reactants)
        
        return has_carbonyl_reactant and has_amine_reactant
        
    def _count_amide_bonds(self, mol) -> int:
        """Count number of amide bonds in molecule."""
        if not mol:
            return 0
            
        amide_pattern = Chem.MolFromSmarts("[C](=[O])[N]")
        return len(mol.GetSubstructMatches(amide_pattern))
