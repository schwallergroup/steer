"""Generated evaluation code for: Late stage alkene formation via Julia-Kocienski olefination"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class JuliaKocienskiOlefination(BaseScoring):
    """
    Evaluates synthesis routes for late-stage alkene formation via Julia-Kocienski olefination.
    Detects when a C=C double bond is formed through this specific reaction type and scores
    based on how late in the synthesis this occurs (later is better).
    """
    
    def __init__(self, config: Dict):
        self.bond_smarts = config["parameters"]["bond_smarts"]  # "C=C"
        self.reaction_type = config["parameters"]["reaction_type"]
        self.timing = config["parameters"]["timing"]  # "late"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Julia-Kocienski olefination doesn't occur
        else:
            # Late-stage olefination is preferred (higher depth fraction is better)
            return 10 * x
            
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents Julia-Kocienski olefination forming a C=C bond.
        """
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        # Split reaction SMILES
        rxn_parts = rxn_smiles.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        products_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        try:
            # Parse molecules
            products = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not products or not all(reactants):
                return False
                
            # Check if a C=C bond is formed
            if not self._alkene_formed(products, reactants):
                return False
                
            # Check for Julia-Kocienski reaction characteristics
            return self._is_julia_kocienski_reaction(products, reactants)
            
        except:
            return False
            
    def _alkene_formed(self, products, reactants) -> bool:
        """Check if a C=C double bond is formed in the reaction."""
        alkene_pattern = Chem.MolFromSmarts(self.bond_smarts)
        if not alkene_pattern:
            return False
            
        # Count alkene bonds in products vs reactants
        product_alkenes = len(products.GetSubstructMatches(alkene_pattern))
        reactant_alkenes = sum(len(r.GetSubstructMatches(alkene_pattern)) for r in reactants)
        
        return product_alkenes > reactant_alkenes
        
    def _is_julia_kocienski_reaction(self, products, reactants) -> bool:
        """
        Check for characteristic patterns of Julia-Kocienski olefination:
        - Presence of sulfone or benzothiazole leaving group patterns
        - Formation of alkene from carbonyl precursor
        """
        # Patterns characteristic of Julia-Kocienski reactions
        sulfone_pattern = Chem.MolFromSmarts("[C][S](=O)(=O)[C]")
        benzothiazole_pattern = Chem.MolFromSmarts("c1ccc2scnc2c1")
        carbonyl_pattern = Chem.MolFromSmarts("[C]=[O]")
        
        # Check if reactants contain Julia-Kocienski reagent patterns
        has_julia_kocienski_reagent = False
        has_carbonyl = False
        
        for reactant in reactants:
            if sulfone_pattern and reactant.HasSubstructMatch(sulfone_pattern):
                has_julia_kocienski_reagent = True
            if benzothiazole_pattern and reactant.HasSubstructMatch(benzothiazole_pattern):
                has_julia_kocienski_reagent = True
            if carbonyl_pattern and reactant.HasSubstructMatch(carbonyl_pattern):
                has_carbonyl = True
                
        return has_julia_kocienski_reagent and has_carbonyl
