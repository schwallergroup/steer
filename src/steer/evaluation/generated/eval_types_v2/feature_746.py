"""Generated evaluation code for: Early pyrrole ring formation via Paal-Knorr synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyPyrroleFormation(BaseScoring):
    """
    Evaluates early pyrrole ring formation via Paal-Knorr synthesis.
    
    The Paal-Knorr synthesis involves condensation of a primary amine with a 
    1,4-dicarbonyl compound to form a pyrrole ring. This class checks for 
    pyrrole formation and rewards earlier occurrence in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.pyrrole_smarts = "[nH]1cccc1"  # pyrrole pattern
        self.dicarbonyl_smarts = "[CX3](=O)[CH2,CH3][CH2,CH3][CX3](=O)"  # 1,4-dicarbonyl pattern
        self.amine_smarts = "[NX3;H2;!$(NC=O)]"  # primary amine pattern
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Pyrrole formation doesn't occur
        else:
            # Early formation is better - invert the depth fraction
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents pyrrole formation via Paal-Knorr synthesis.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        # Parse molecules
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
        except:
            return False
        
        # Check if pyrrole is formed in the product
        pyrrole_pattern = Chem.MolFromSmarts(self.pyrrole_smarts)
        if not product.HasSubstructMatch(pyrrole_pattern):
            return False
            
        # Check if reactants contain the expected Paal-Knorr components
        dicarbonyl_pattern = Chem.MolFromSmarts(self.dicarbonyl_smarts)
        amine_pattern = Chem.MolFromSmarts(self.amine_smarts)
        
        has_dicarbonyl = False
        has_amine = False
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(dicarbonyl_pattern):
                has_dicarbonyl = True
            if reactant.HasSubstructMatch(amine_pattern):
                has_amine = True
                
        # Verify that reactants don't already contain pyrrole
        reactants_have_pyrrole = any(r.HasSubstructMatch(pyrrole_pattern) for r in reactants)
        
        # This is a Paal-Knorr pyrrole formation if:
        # 1. Product contains pyrrole
        # 2. Reactants contain dicarbonyl and amine
        # 3. Reactants don't already contain pyrrole
        return has_dicarbonyl and has_amine and not reactants_have_pyrrole
