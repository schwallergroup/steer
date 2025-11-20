"""Generated evaluation code for: Early thiazole ring formation via Hantzsch synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyThiazoleHantzsch(BaseScoring):
    """
    Evaluates if thiazole ring formation occurs early in the synthesis route 
    using Hantzsch synthesis methodology.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1scnc1"
        self.timing = config["parameters"]["timing"]  # "early"
        self.thiazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Thiazole formation doesn't happen
        else:
            if self.timing == "early":
                return 1 - x  # Earlier formation is better (closer to 1.0)
            else:
                return x  # Later formation is better (closer to 1.0)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents thiazole ring formation
        via Hantzsch synthesis.
        """
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        # Split reaction into reactants and products
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        # Parse molecules
        try:
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
        except:
            return False
            
        if not reactants or not products:
            return False
        
        # Check if thiazole ring is formed (present in products but not reactants)
        reactants_have_thiazole = any(mol.HasSubstructMatch(self.thiazole_pattern) for mol in reactants)
        products_have_thiazole = any(mol.HasSubstructMatch(self.thiazole_pattern) for mol in products)
        
        # Thiazole must be formed in this step
        if not products_have_thiazole or reactants_have_thiazole:
            return False
            
        # Check for Hantzsch synthesis pattern
        return self._is_hantzsch_synthesis(reactants, products)
    
    def _is_hantzsch_synthesis(self, reactants, products) -> bool:
        """
        Check if the reaction follows Hantzsch thiazole synthesis pattern.
        Typically involves α-haloketone + thioamide or similar.
        """
        # Pattern for α-haloketone (C(=O)C[Cl,Br,I])
        haloketone_pattern = Chem.MolFromSmarts("[#6](=[#8])[#6][Cl,Br,I]")
        
        # Pattern for thioamide (C(=S)N)
        thioamide_pattern = Chem.MolFromSmarts("[#6](=[#16])[#7]")
        
        # Alternative pattern for thiourea derivatives
        thiourea_pattern = Chem.MolFromSmarts("[#7][#6](=[#16])[#7]")
        
        # Check if we have the typical Hantzsch reactants
        has_haloketone = any(mol.HasSubstructMatch(haloketone_pattern) for mol in reactants)
        has_thioamide = any(mol.HasSubstructMatch(thioamide_pattern) for mol in reactants)
        has_thiourea = any(mol.HasSubstructMatch(thiourea_pattern) for mol in reactants)
        
        # Hantzsch synthesis typically requires haloketone + thioamide/thiourea
        return has_haloketone and (has_thioamide or has_thiourea)
