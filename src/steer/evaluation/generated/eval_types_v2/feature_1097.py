"""Generated evaluation code for: Late stage cyclopropanation via carbene insertion"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCyclopropanation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage cyclopropanation via carbene insertion.
    Detects formation of cyclopropane rings through carbene insertion reactions and
    rewards routes where this occurs at later stages (closer to the target molecule).
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "C1CC1"
        self.timing = config["parameters"]["timing"]  # "late"
        self.formation_method = config["parameters"]["formation_method"]  # "carbene_insertion"
        self.cyclopropane_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
        # SMARTS pattern for diazo compounds (common carbene precursors)
        self.diazo_pattern = Chem.MolFromSmarts("[C]=[N+]=[N-]")
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        For late-stage reactions, higher depth fractions (closer to target) get higher scores.
        """
        if x < 0:
            return 0  # Cyclopropanation doesn't occur
        
        if self.timing == "late":
            # Late-stage: reward higher depth fractions (x closer to 1)
            return x * 10
        else:
            # Early-stage: reward lower depth fractions (x closer to 0)
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction node represents cyclopropanation via carbene insertion.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if cyclopropane ring is formed (present in product but not in reactants)
            product_has_cyclopropane = product_mol.HasSubstructMatch(self.cyclopropane_pattern)
            
            if not product_has_cyclopropane:
                return False
            
            # Check if any reactant already has cyclopropane (would indicate it's not formation)
            reactants_have_cyclopropane = any(mol.HasSubstructMatch(self.cyclopropane_pattern) 
                                            for mol in reactant_mols)
            
            if reactants_have_cyclopropane:
                return False
            
            # Check for carbene insertion signature: presence of diazo compound in reactants
            if self.formation_method == "carbene_insertion":
                has_diazo_reactant = any(mol.HasSubstructMatch(self.diazo_pattern) 
                                       for mol in reactant_mols)
                
                # Alternative check: look for common carbene insertion patterns
                # Check for alkene + carbene precursor pattern
                alkene_pattern = Chem.MolFromSmarts("C=C")
                has_alkene_reactant = any(mol.HasSubstructMatch(alkene_pattern) 
                                        for mol in reactant_mols)
                
                return has_diazo_reactant and has_alkene_reactant
            
            # If formation_method is not specified, just check for cyclopropane formation
            return True
            
        except Exception:
            return False
