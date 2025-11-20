"""Generated evaluation code for: Late stage benzyl ether deprotection"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylEtherDeprotection(BaseScoring):
    """
    Evaluates benzyl ether deprotection timing in synthesis routes.
    Checks if benzyl ether deprotection occurs as the final step,
    particularly when alkenes are present in the substrate.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "final_step")
        self.check_alkenes = config.get("substrate_alkenes", True)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Deprotection doesn't happen
        
        if self.timing == "final_step":
            # Reward deprotection very close to the end (depth fraction near 0)
            if x <= 0.1:  # Final 10% of route
                return 10
            elif x <= 0.3:  # Final 30% of route
                return 7
            else:
                return max(0, 5 - x * 5)  # Penalty for earlier deprotection
        
        return 1 - x  # General preference for late-stage
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves benzyl ether deprotection"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants, products = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Benzyl ether pattern: aromatic ring connected to OCH2-
            benzyl_ether_pattern = Chem.MolFromSmarts("[cH0,cH1:1][CH2:2][O:3]")
            # Benzyl alcohol/toluene as leaving group
            benzyl_leaving_pattern = Chem.MolFromSmarts("c[CH2][OH]")
            toluene_pattern = Chem.MolFromSmarts("c1ccccc1[CH3]")
            
            # Check if reactant has benzyl ether and products have benzyl alcohol/toluene
            has_benzyl_ether_reactant = any(mol.HasSubstructMatch(benzyl_ether_pattern) 
                                          for mol in reactant_mols)
            
            has_benzyl_leaving = any(mol.HasSubstructMatch(benzyl_leaving_pattern) or 
                                   mol.HasSubstructMatch(toluene_pattern)
                                   for mol in product_mols)
            
            if not (has_benzyl_ether_reactant and has_benzyl_leaving):
                return False
            
            # Additional check for alkenes if specified
            if self.check_alkenes:
                alkene_pattern = Chem.MolFromSmarts("[CH,CH2]=[CH,CH2]")
                has_alkenes = any(mol.HasSubstructMatch(alkene_pattern) 
                                for mol in reactant_mols + product_mols)
                return has_alkenes
            
            return True
            
        except Exception:
            return False
