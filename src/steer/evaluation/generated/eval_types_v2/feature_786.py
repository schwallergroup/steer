"""Generated evaluation code for: Convergent synthesis via amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentAmideCoupling(BaseScoring):
    """
    Evaluates convergent synthesis routes that use amide coupling reactions.
    Checks if an amide formation reaction occurs at a specified stage with
    two fragments of sufficient complexity.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_stage = config.get("coupling_stage", "late")  # "early", "mid", "late"
        self.min_fragment_size = config.get("min_fragment_size", 8)  # minimum heavy atoms per fragment
        
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score. For convergent synthesis, earlier is often better."""
        if x < 0:
            return 0  # No amide coupling found
        
        if self.coupling_stage == "early":
            return 1 - x  # Earlier coupling preferred
        elif self.coupling_stage == "late":
            return x  # Later coupling preferred
        else:  # "mid"
            return 1 - abs(x - 0.5)  # Middle coupling preferred
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents a convergent amide coupling."""
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles")
        
        if not rxn_smiles:
            return False
            
        try:
            # Split reaction SMILES
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            reactants = reactants_smiles.split(".")
            
            # Check if we have the expected number of fragments
            if len(reactants) != self.fragment_count:
                return False
            
            # Parse molecules
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants]
            
            if not product_mol or any(mol is None for mol in reactant_mols):
                return False
            
            # Check if reactants meet minimum size requirements
            for mol in reactant_mols:
                if mol.GetNumHeavyAtoms() < self.min_fragment_size:
                    return False
            
            # Check for amide formation
            if self._is_amide_formation(reactant_mols, product_mol):
                return True
                
        except Exception:
            return False
            
        return False
    
    def _is_amide_formation(self, reactants, product) -> bool:
        """Check if reaction involves amide bond formation."""
        # Define patterns for amide formation reactants
        carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")
        amine_pattern = Chem.MolFromSmarts("[N;H1,H2]")
        acid_chloride_pattern = Chem.MolFromSmarts("[C](=O)[Cl]")
        ester_pattern = Chem.MolFromSmarts("[C](=O)[O][C]")
        
        # Pattern for amide product
        amide_pattern = Chem.MolFromSmarts("[C](=O)[N]")
        
        # Check if product contains amide
        if not product.HasSubstructMatch(amide_pattern):
            return False
        
        # Check reactants for typical amide formation partners
        has_carbonyl_partner = False
        has_amine = False
        
        for reactant in reactants:
            if (reactant.HasSubstructMatch(carboxylic_acid_pattern) or 
                reactant.HasSubstructMatch(acid_chloride_pattern) or
                reactant.HasSubstructMatch(ester_pattern)):
                has_carbonyl_partner = True
            
            if reactant.HasSubstructMatch(amine_pattern):
                has_amine = True
        
        return has_carbonyl_partner and has_amine
