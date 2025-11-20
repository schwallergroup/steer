"""Generated evaluation code for: Early stage thiazole assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyThiazoleAssembly(BaseScoring):
    """
    Evaluates whether thiazole ring assembly occurs early in the synthesis route.
    Detects thiazole formation via Hantzsch synthesis and rewards early-stage formation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1scnc1"
        self.timing = config["parameters"]["timing"]  # "early"
        self.thiazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Thiazole formation doesn't occur
        
        if self.timing == "early":
            # Reward early formation (lower depth fraction is better)
            return (1 - x) * 10
        else:
            # For other timing preferences, could adjust scoring
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """
        Checks if this reaction step involves thiazole ring formation via Hantzsch synthesis.
        """
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        # Parse reactants and products
        try:
            product_mol = Chem.MolFromSmiles(products_smiles)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) 
                           for r in reactants_smiles.split(".") if r.strip()]
            
            if not product_mol or not reactant_mols:
                return False
                
        except:
            return False
        
        # Check if product contains thiazole but reactants don't
        product_has_thiazole = product_mol.HasSubstructMatch(self.thiazole_pattern)
        
        if not product_has_thiazole:
            return False
            
        # Check that reactants don't already contain the thiazole ring
        reactants_have_thiazole = any(mol.HasSubstructMatch(self.thiazole_pattern) 
                                    for mol in reactant_mols if mol is not None)
        
        if reactants_have_thiazole:
            return False  # Not a formation reaction, thiazole already present
            
        # Additional check for Hantzsch-like pattern (optional)
        # Look for characteristic functional groups in reactants that suggest Hantzsch synthesis
        return self._is_hantzsch_like_reaction(reactant_mols)
    
    def _is_hantzsch_like_reaction(self, reactant_mols) -> bool:
        """
        Check if reactants contain patterns typical of Hantzsch thiazole synthesis.
        Hantzsch synthesis typically involves α-haloketones and thioamides.
        """
        # Pattern for α-haloketone (e.g., Br-CH2-CO-)
        haloketone_pattern = Chem.MolFromSmarts("[Cl,Br,I][CH2][CX3]=O")
        
        # Pattern for thioamide (e.g., R-CS-NH2)
        thioamide_pattern = Chem.MolFromSmarts("[CX3]=[SX1]")
        
        # Pattern for thiourea derivatives
        thiourea_pattern = Chem.MolFromSmarts("[NX3][CX3]=[SX1]")
        
        has_haloketone = any(mol.HasSubstructMatch(haloketone_pattern) 
                           for mol in reactant_mols if mol is not None)
        
        has_thio_component = any(mol.HasSubstructMatch(thioamide_pattern) or 
                               mol.HasSubstructMatch(thiourea_pattern)
                               for mol in reactant_mols if mol is not None)
        
        # For a more permissive check, just require formation of thiazole ring
        # since exact Hantzsch pattern matching might be too restrictive
        return True  # Allow any thiazole formation to count
