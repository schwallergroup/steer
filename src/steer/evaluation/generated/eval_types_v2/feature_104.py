"""Generated evaluation code for: Late stage epoxide ring opening"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EpoxideRingOpening(BaseScoring):
    """
    Evaluates whether epoxide ring opening occurs at the desired timing in the synthesis route.
    Detects epoxide (3-membered oxygen-containing ring) breaking reactions and scores based on
    when they occur in the route (late stage preferred).
    """
    
    def __init__(self, config: Dict):
        self.epoxide_smarts = config.get("ring_smarts", "C1OC1")
        self.timing = config.get("timing", "late")  # "early", "late", or specific depth
        self.direction = config.get("direction", "breaking")  # "breaking" or "forming"
        
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10 scale)"""
        if x < 0:
            return 0  # Epoxide ring opening doesn't occur
        
        if self.timing == "late":
            # Late stage preferred - higher score for larger depth fractions
            return 10 * x
        elif self.timing == "early":
            # Early stage preferred - higher score for smaller depth fractions  
            return 10 * (1 - x)
        else:
            # Specific timing target
            target_depth = float(self.timing)
            return max(0, 10 - 10 * abs(x - target_depth))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves epoxide ring opening/forming"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse molecules
            reactants = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol:
                    reactants.append(mol)
                    
            products = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol:
                    products.append(mol)
            
            if not reactants or not products:
                return False
                
            # Create epoxide pattern matcher
            epoxide_pattern = Chem.MolFromSmarts(self.epoxide_smarts)
            if not epoxide_pattern:
                return False
                
            # Count epoxides in reactants and products
            reactant_epoxides = sum(len(mol.GetSubstructMatches(epoxide_pattern)) 
                                  for mol in reactants)
            product_epoxides = sum(len(mol.GetSubstructMatches(epoxide_pattern)) 
                                 for mol in products)
            
            if self.direction == "breaking":
                # Epoxide ring opening: more epoxides in reactants than products
                return reactant_epoxides > product_epoxides
            else:  # "forming"
                # Epoxide ring formation: more epoxides in products than reactants
                return product_epoxides > reactant_epoxides
                
        except Exception:
            return False
