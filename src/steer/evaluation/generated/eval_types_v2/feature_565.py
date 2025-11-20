"""Generated evaluation code for: Initial heterocycle formation via cyclization condensation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PyridazinoneFormationFirst(BaseScoring):
    """
    Evaluates whether pyridazinone ring formation occurs early in the synthesis
    via intramolecular cyclization condensation.
    
    Checks for the formation of pyridazinone (six-membered ring with two adjacent nitrogens
    and a ketone) and scores based on how early this occurs in the route.
    """
    
    def __init__(self, config: Dict):
        self.target_timing = config["parameters"]["timing"]  # "first"
        self.method = config["parameters"]["method"]  # "intramolecular_condensation"
        # SMARTS pattern for pyridazinone core (6-membered ring, adjacent N-N, ketone)
        self.pyridazinone_pattern = Chem.MolFromSmarts("[#6]1[#6][#6][#6](=[#8])[#7][#7]1")
    
    def route_scoring(self, x) -> float:
        """
        Score based on timing of pyridazinone formation.
        Early formation (low depth fraction) gets higher scores.
        """
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.target_timing == "first":
            # Reward very early formation, penalize late formation
            if x <= 0.2:  # First 20% of route
                return 10
            elif x <= 0.4:  # First 40% of route  
                return 8
            elif x <= 0.6:  # First 60% of route
                return 6
            elif x <= 0.8:  # First 80% of route
                return 3
            else:  # Late formation
                return 1
        
        return 0
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents pyridazinone formation
        via intramolecular cyclization condensation.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check if pyridazinone ring is formed in this step
            reactants_have_ring = any(mol.HasSubstructMatch(self.pyridazinone_pattern) for mol in reactants)
            products_have_ring = any(mol.HasSubstructMatch(self.pyridazinone_pattern) for mol in products)
            
            # Ring must be formed (not present in reactants, present in products)
            if reactants_have_ring or not products_have_ring:
                return False
            
            # Check for intramolecular condensation characteristics
            if self.method == "intramolecular_condensation":
                return self._is_intramolecular_condensation(reactants, products)
            
            return True
            
        except Exception:
            return False
    
    def _is_intramolecular_condensation(self, reactants, products) -> bool:
        """
        Check if the reaction represents an intramolecular condensation.
        Characteristics: single main reactant, water elimination typical.
        """
        # Filter out small molecules (likely catalysts/reagents)
        main_reactants = [mol for mol in reactants if mol.GetNumAtoms() > 5]
        main_products = [mol for mol in products if mol.GetNumAtoms() > 5]
        
        # Should have one main reactant forming one main product (intramolecular)
        if len(main_reactants) != 1 or len(main_products) != 1:
            return False
        
        reactant = main_reactants[0]
        product = main_products[0]
        
        # Intramolecular cyclization: reactant should have appropriate precursor groups
        # Look for potential condensation partners (e.g., carbonyl + hydrazine/amine)
        carbonyl_pattern = Chem.MolFromSmarts("[#6]=[#8]")
        hydrazine_pattern = Chem.MolFromSmarts("[#7][#7]")
        
        has_carbonyl = reactant.HasSubstructMatch(carbonyl_pattern)
        has_hydrazine = reactant.HasSubstructMatch(hydrazine_pattern)
        
        # Should have both functional groups in reactant for condensation
        return has_carbonyl and has_hydrazine
