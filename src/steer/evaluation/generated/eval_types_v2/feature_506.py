"""Generated evaluation code for: Late stage Williamson ether synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageWilliamsonEther(BaseScoring):
    """
    Evaluates whether a Williamson ether synthesis occurs at a late stage in the route.
    Williamson ether synthesis involves nucleophilic substitution of an alkoxide with
    an alkyl halide or similar electrophile to form an ether bond.
    """
    
    def __init__(self, config: Dict):
        self.step_position = config.get("step_position", 1)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Williamson ether synthesis doesn't occur
        else:
            return 1 - x  # Late-stage reaction is better (lower depth fraction)
    
    def hit_condition(self, d) -> bool:
        """
        Detects Williamson ether synthesis by checking for:
        1. Formation of new C-O-C ether bond
        2. Breaking of C-X bond (X = halide or similar leaving group)
        3. Presence of alkoxide nucleophile pattern
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for ether formation pattern
            return self._detect_williamson_ether_formation(reactants, products)
            
        except Exception:
            return False
    
    def _detect_williamson_ether_formation(self, reactants, products) -> bool:
        """
        Detects Williamson ether synthesis by checking for characteristic patterns:
        - Alkyl halide (R-X where X = Cl, Br, I)
        - Alkoxide or phenoxide nucleophile
        - Formation of new ether bond (C-O-C)
        """
        # Patterns for alkyl/aryl halides
        halide_patterns = [
            Chem.MolFromSmarts("[C,c][Cl,Br,I]"),  # Alkyl/aryl halides
            Chem.MolFromSmarts("[C,c]OS(=O)(=O)[C,c]"),  # Tosylates
            Chem.MolFromSmarts("[C,c]OS(=O)(=O)C(F)(F)F")  # Triflates
        ]
        
        # Pattern for alkoxide/phenoxide nucleophile (often as salt)
        alkoxide_patterns = [
            Chem.MolFromSmarts("[C,c][O-]"),  # Alkoxide anion
            Chem.MolFromSmarts("[C,c]O"),     # Alcohol (can be deprotonated in situ)
        ]
        
        # Check reactants for halide and alkoxide patterns
        has_halide = False
        has_alkoxide = False
        
        for reactant in reactants:
            # Check for halide pattern
            for halide_pattern in halide_patterns:
                if reactant.HasSubstructMatch(halide_pattern):
                    has_halide = True
                    break
                    
            # Check for alkoxide pattern
            for alkoxide_pattern in alkoxide_patterns:
                if reactant.HasSubstructMatch(alkoxide_pattern):
                    has_alkoxide = True
                    break
        
        # Check for ether formation in products
        ether_pattern = Chem.MolFromSmarts("[C,c]O[C,c]")
        has_ether_product = any(product.HasSubstructMatch(ether_pattern) for product in products)
        
        # Additional check: verify net C-O bond formation
        # Count C-O-C ethers in reactants vs products
        reactant_ethers = sum(len(mol.GetSubstructMatches(ether_pattern)) for mol in reactants)
        product_ethers = sum(len(mol.GetSubstructMatches(ether_pattern)) for mol in products)
        
        has_new_ether = product_ethers > reactant_ethers
        
        return has_halide and has_alkoxide and has_ether_product and has_new_ether
