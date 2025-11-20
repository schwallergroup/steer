"""Generated evaluation code for: Late stage Cadogan cyclization for carbazole formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CadoganCyclization(BaseScoring):
    """
    Evaluates synthesis routes based on late-stage Cadogan cyclization for carbazole formation.
    
    The Cadogan cyclization involves nitrene insertion from an aryl azide intermediate to form
    a carbazole-like core structure. This class checks for the formation of the carbazole
    pattern and rewards routes where this occurs late in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.carbazole_pattern = Chem.MolFromSmarts(config["parameters"]["ring_smarts"])
        self.timing = config["parameters"]["timing"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Carbazole formation doesn't happen
        else:
            # Late-stage formation is better, so higher depth fraction gets higher score
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms a carbazole core via Cadogan cyclization.
        """
        # Get reaction SMILES
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        reactants_smiles, product_smiles = rxn_smiles.split(">>")
        
        # Parse molecules
        product = Chem.MolFromSmiles(product_smiles)
        if not product:
            return False
            
        reactants = []
        for r_smiles in reactants_smiles.split("."):
            mol = Chem.MolFromSmiles(r_smiles)
            if mol:
                reactants.append(mol)
        
        if not reactants:
            return False
            
        # Check if product contains carbazole pattern
        if not product.HasSubstructMatch(self.carbazole_pattern):
            return False
            
        # Check if any reactant already contains the carbazole pattern
        # (if so, this isn't a carbazole-forming reaction)
        for reactant in reactants:
            if reactant.HasSubstructMatch(self.carbazole_pattern):
                return False
                
        # Check for characteristic Cadogan cyclization pattern:
        # Should involve azide reduction and cyclization
        azide_pattern = Chem.MolFromSmarts("N=[N+]=[N-]")  # Azide group
        nitrene_precursor_pattern = Chem.MolFromSmarts("c-N=[N+]=[N-]")  # Aryl azide
        
        # Look for aryl azide in reactants
        has_aryl_azide = any(reactant.HasSubstructMatch(nitrene_precursor_pattern) 
                           for reactant in reactants)
        
        return has_aryl_azide
