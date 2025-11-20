"""Generated evaluation code for: Grignard addition to N-Boc-4-piperidone"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class GrignardAdditionToPiperidone(BaseScoring):
    """
    Evaluates synthesis routes for Grignard addition to N-Boc-4-piperidone.
    Rewards late-stage organometallic C-C bond formation on protected piperidone scaffold.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "late")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur in route
        
        if self.timing == "late":
            return 1 - x  # Later is better (higher score for smaller depth fraction)
        elif self.timing == "early":
            return x  # Earlier is better (higher score for larger depth fraction)
        else:
            return 1 if x >= 0 else 0  # Just presence/absence
    
    def hit_condition(self, d) -> bool:
        """Check if reaction is Grignard addition to N-Boc-4-piperidone"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and product
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for N-Boc-4-piperidone electrophile in reactants
            boc_piperidone_pattern = Chem.MolFromSmarts("[#6]1[#6][#7]([#6](=[#8])[#8][#6]([#6])([#6])[#6])[#6][#6]1=[#8]")  # N-Boc-4-piperidone core
            has_boc_piperidone = any(mol.HasSubstructMatch(boc_piperidone_pattern) for mol in reactants)
            
            if not has_boc_piperidone:
                return False
            
            # Check for Grignard reagent pattern (C-MgX where X is halogen)
            grignard_pattern = Chem.MolFromSmarts("[#6]-[Mg]-[F,Cl,Br,I]")
            has_grignard = any(mol.HasSubstructMatch(grignard_pattern) for mol in reactants)
            
            # Also check for organolithium as alternative (C-Li)
            organolithium_pattern = Chem.MolFromSmarts("[#6]-[Li]")
            has_organolithium = any(mol.HasSubstructMatch(organolithium_pattern) for mol in reactants)
            
            if not (has_grignard or has_organolithium):
                return False
            
            # Check product has tertiary alcohol at position 4 of piperidine
            # (carbonyl should be converted to tertiary alcohol)
            tertiary_alcohol_piperidine = Chem.MolFromSmarts("[#6]1[#6][#7]([#6](=[#8])[#8][#6]([#6])([#6])[#6])[#6][#6]1([#8])[#6]")
            has_product_structure = product.HasSubstructMatch(tertiary_alcohol_piperidine)
            
            return has_product_structure
            
        except Exception:
            return False
