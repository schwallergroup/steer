"""Generated evaluation code for: Grignard addition to aldehyde with ketone present"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class GrignardAldehydeKetoneSelectivity(BaseScoring):
    """
    Evaluates routes for Grignard addition to aldehyde in the presence of ketone.
    Checks for chemoselectivity challenge where Grignard reagent must selectively
    react with aldehyde over competing ketone electrophile.
    """
    
    def __init__(self, config: Dict):
        self.chemoselectivity_required = config.get("chemoselectivity_challenge", True)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            return 1 - x  # Earlier occurrence is better for selectivity demonstration
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Grignard addition to aldehyde with ketone present"""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            product = Chem.MolFromSmiles(rxn[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            # Check for Grignard reagent (R-MgX pattern)
            grignard_pattern = Chem.MolFromSmarts("[C,c]-[Mg]-[Cl,Br,I]")
            has_grignard = any(mol.HasSubstructMatch(grignard_pattern) for mol in reactants)
            
            if not has_grignard:
                return False
            
            # Check for aldehyde in reactants
            aldehyde_pattern = Chem.MolFromSmarts("[C;H1](=O)")
            aldehyde_reactant = None
            for mol in reactants:
                if mol.HasSubstructMatch(aldehyde_pattern):
                    aldehyde_reactant = mol
                    break
            
            if not aldehyde_reactant:
                return False
            
            # Check for ketone in the same molecule as aldehyde (chemoselectivity challenge)
            ketone_pattern = Chem.MolFromSmarts("[C;H0](=O)[C,c]")
            has_ketone_in_substrate = aldehyde_reactant.HasSubstructMatch(ketone_pattern)
            
            if not has_ketone_in_substrate and self.chemoselectivity_required:
                return False
            
            # Verify product has secondary alcohol (from aldehyde addition)
            # and ketone remains unreacted
            secondary_alcohol_pattern = Chem.MolFromSmarts("[C;H1]([OH])")
            has_sec_alcohol = product.HasSubstructMatch(secondary_alcohol_pattern)
            
            # Check that ketone is still present in product (unreacted)
            ketone_still_present = product.HasSubstructMatch(ketone_pattern)
            
            return has_sec_alcohol and ketone_still_present
            
        except Exception:
            return False
