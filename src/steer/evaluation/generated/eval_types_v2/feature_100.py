"""Generated evaluation code for: Late isoxazole ring formation via cycloaddition"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateIsoxazoleFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage isoxazole ring formation via cycloaddition.
    Rewards routes where isoxazole rings are formed later in the synthesis through
    [3+2] cycloaddition reactions between nitrile oxides and alkynes.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.formation_method = config["parameters"]["formation_method"]
        self.timing = config["parameters"]["timing"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        """
        Converts depth fraction to score (0-10).
        Later formation (higher x) gets better score for late timing preference.
        """
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # Late-stage formation is better, so higher depth fraction = higher score
            return x * 10
            
    def hit_condition(self, d) -> bool:
        """
        Checks if this reaction node represents isoxazole formation via cycloaddition.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        # Parse molecules
        try:
            product_mol = Chem.MolFromSmiles(products_smiles)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
                
        except:
            return False
            
        # Check if product contains isoxazole ring
        if not product_mol.HasSubstructMatch(self.ring_pattern):
            return False
            
        # Check if reactants lack the isoxazole ring (ring formation)
        reactants_have_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactant_mols)
        if reactants_have_ring:
            return False  # Ring already exists, not formation
            
        # Check for cycloaddition pattern: should have nitrile oxide and alkyne precursors
        if self.formation_method == "cycloaddition":
            return self._is_cycloaddition_pattern(reactant_mols, product_mol)
            
        return True
        
    def _is_cycloaddition_pattern(self, reactants, product) -> bool:
        """
        Checks if the reaction pattern matches [3+2] cycloaddition for isoxazole formation.
        Looks for nitrile oxide (C#N-O) and alkyne (C#C) patterns in reactants.
        """
        nitrile_oxide_pattern = Chem.MolFromSmarts("[C-]#[N+][O-]")  # Nitrile oxide
        alkyne_pattern = Chem.MolFromSmarts("C#C")  # Alkyne
        
        has_nitrile_oxide = any(mol.HasSubstructMatch(nitrile_oxide_pattern) for mol in reactants)
        has_alkyne = any(mol.HasSubstructMatch(alkyne_pattern) for mol in reactants)
        
        # Alternative nitrile oxide patterns (different representations)
        if not has_nitrile_oxide:
            alt_nitrile_oxide = Chem.MolFromSmarts("C#[N+][O-]")
            has_nitrile_oxide = any(mol.HasSubstructMatch(alt_nitrile_oxide) for mol in reactants)
            
        return has_nitrile_oxide and has_alkyne
