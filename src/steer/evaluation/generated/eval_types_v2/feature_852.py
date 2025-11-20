"""Generated evaluation code for: Late thiazole ring formation via Hantzsch synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateThiazoleHantzschFormation(BaseScoring):
    """
    Evaluates late-stage thiazole ring formation via Hantzsch synthesis.
    Checks for thiazole formation from thioamide and alpha-halo carbonyl components.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1scnc1"
        self.formation_method = config["parameters"]["formation_method"]  # "hantzsch"
        self.stage = config["parameters"]["stage"]  # "late"
        
        # Thiazole SMARTS pattern
        self.thiazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
        # Hantzsch synthesis component patterns
        self.thioamide_pattern = Chem.MolFromSmarts("[C,c][C](=[S])[NH2,NH1]")
        self.halo_carbonyl_pattern = Chem.MolFromSmarts("[Cl,Br,I][CH2][C](=O)")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Thiazole formation doesn't occur
        else:
            # Late-stage formation is better (higher depth fraction gives higher score)
            return 10 * x  # Scale to 0-10 range
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves thiazole formation via Hantzsch synthesis
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            # Parse molecules
            products = [Chem.MolFromSmiles(products_smiles)]
            reactants = []
            for r_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles.strip())
                if mol:
                    reactants.append(mol)
            
            # Check if product contains thiazole ring
            product_has_thiazole = any(
                mol and mol.HasSubstructMatch(self.thiazole_pattern) 
                for mol in products if mol
            )
            
            if not product_has_thiazole:
                return False
            
            # Check if reactants contain Hantzsch components
            has_thioamide = any(
                mol.HasSubstructMatch(self.thioamide_pattern) 
                for mol in reactants
            )
            
            has_halo_carbonyl = any(
                mol.HasSubstructMatch(self.halo_carbonyl_pattern) 
                for mol in reactants
            )
            
            # Additional check: reactants should NOT contain thiazole
            reactants_have_thiazole = any(
                mol.HasSubstructMatch(self.thiazole_pattern) 
                for mol in reactants
            )
            
            # True Hantzsch thiazole formation: product has thiazole, 
            # reactants have required components but no thiazole
            return (product_has_thiazole and 
                    has_thioamide and 
                    has_halo_carbonyl and 
                    not reactants_have_thiazole)
                    
        except Exception:
            return False
