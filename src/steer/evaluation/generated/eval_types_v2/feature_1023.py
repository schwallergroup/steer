"""Generated evaluation code for: Early nitrene-mediated azaindole core formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class NitreneAzaindoleFormation(BaseScoring):
    """
    Evaluates early nitrene-mediated azaindole core formation in synthesis routes.
    
    This class checks if the tricyclic azaindole core (c1cnc2[nH]ccc21) is formed
    early in the synthesis route through nitrene insertion chemistry, typically
    involving azide decomposition and C-H insertion mechanisms.
    """
    
    def __init__(self, config: Dict):
        self.azaindole_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.mechanism = config["parameters"]["mechanism"]
        
        # Compile the azaindole pattern
        self.azaindole_pattern = Chem.MolFromSmarts(self.azaindole_smarts)
        
        # Nitrene precursor patterns (azides and related structures)
        self.nitrene_patterns = [
            Chem.MolFromSmarts("[N-]=[N+]=[N-]"),  # Azide group
            Chem.MolFromSmarts("c[N-]=[N+]=[N-]"),  # Aryl azide
            Chem.MolFromSmarts("[NH2+][N-][N-]"),   # Protonated azide
            Chem.MolFromSmarts("N=[N+]=[N-]")       # Alternative azide form
        ]
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10), favoring early formation."""
        if x < 0:
            return 0  # Reaction doesn't occur
        
        if self.timing == "early":
            # Early formation is better - higher score for lower depth
            return max(0, 10 * (1 - x))
        else:
            # If not specifically early, just check if it happens
            return 5.0
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents nitrene-mediated azaindole formation."""
        metadata = d.get("metadata", {})
        
        # Check if we have the required reaction data
        if "mapped_reaction_smiles" not in metadata:
            return False
        
        try:
            rxn_smiles = metadata["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check if azaindole core is formed (present in products but not in reactants)
            azaindole_in_products = any(mol.HasSubstructMatch(self.azaindole_pattern) for mol in products)
            azaindole_in_reactants = any(mol.HasSubstructMatch(self.azaindole_pattern) for mol in reactants)
            
            # Must form the azaindole core in this step
            if not azaindole_in_products or azaindole_in_reactants:
                return False
            
            # Check for nitrene precursors in reactants
            has_nitrene_precursor = False
            for reactant in reactants:
                for pattern in self.nitrene_patterns:
                    if reactant.HasSubstructMatch(pattern):
                        has_nitrene_precursor = True
                        break
                if has_nitrene_precursor:
                    break
            
            # Check for additional indicators of nitrene chemistry
            # Look for loss of N2 (common in azide decomposition)
            reactant_atoms = sum(mol.GetNumAtoms() for mol in reactants)
            product_atoms = sum(mol.GetNumAtoms() for mol in products)
            
            # Typical nitrene insertion loses N2 (2 atoms)
            nitrogen_loss = reactant_atoms - product_atoms >= 2
            
            # Check if any products contain molecular nitrogen
            has_n2_product = any(Chem.MolToSmiles(mol) == "N#N" for mol in products)
            
            return (has_nitrene_precursor and (nitrogen_loss or has_n2_product))
            
        except Exception:
            # If there's any error in parsing, return False
            return False
