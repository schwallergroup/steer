"""Generated evaluation code for: Early indole ring formation via reductive cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyIndoleFormation(BaseScoring):
    """
    Evaluates whether indole ring formation occurs early in the synthesis route
    via reductive cyclization. Checks for the formation of the indole core structure
    and rewards earlier occurrence in the synthetic sequence.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
        # Pattern for reductive cyclization precursor (ortho-nitroaryl)
        self.precursor_pattern = Chem.MolFromSmarts("c1ccc(N(=O)=O)c(*)c1")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "early":
            return 1 - x  # Earlier formation is better (higher score)
        elif self.timing == "late":
            return x  # Later formation is better
        else:
            return 0.5  # Neutral if timing preference not specified
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves indole ring formation
        via reductive cyclization
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        reactants_smiles, products_smiles = mapped_rxn.split(">>")
        
        # Parse reactants and products
        try:
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) 
                           for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) 
                          for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
        except:
            return False
        
        if not reactant_mols or not product_mols:
            return False
        
        # Check for indole formation: absent in reactants, present in products
        has_indole_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) 
                                    for mol in reactant_mols)
        has_indole_in_products = any(mol.HasSubstructMatch(self.ring_pattern) 
                                   for mol in product_mols)
        
        # Check for reductive cyclization pattern
        has_precursor_in_reactants = any(mol.HasSubstructMatch(self.precursor_pattern) 
                                       for mol in reactant_mols)
        
        # Must be ring formation (not breaking) and involve reductive cyclization
        if self.direction == "formation":
            ring_formed = not has_indole_in_reactants and has_indole_in_products
            is_reductive_cyclization = has_precursor_in_reactants and not any(
                mol.HasSubstructMatch(self.precursor_pattern) for mol in product_mols
            )
            
            return ring_formed and is_reductive_cyclization
        
        return False
