"""Generated evaluation code for: Late stage cyclopropane ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CyclopropaneFormation(BaseScoring):
    """
    Evaluates late-stage cyclopropane ring formation in synthesis routes.
    Detects when cyclopropane rings are formed (not present in reactants but present in products)
    and scores based on timing preference for late-stage formation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config.get("ring_smarts", "C1CC1")
        self.timing = config.get("timing", "late")  # "early", "late", or "any"
        self.direction = config.get("direction", "formation")  # "formation" or "breaking"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cyclopropane formation doesn't occur
        
        if self.timing == "late":
            return 1 - x  # Later formation is better (lower depth fraction is worse)
        elif self.timing == "early":
            return x  # Earlier formation is better (higher depth fraction is worse)
        else:  # "any"
            return 1  # Just presence matters, not timing
    
    def hit_condition(self, d) -> bool:
        """
        Check if cyclopropane ring formation occurs in this reaction step.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactant_mols = []
            for r_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles.strip())
                if mol:
                    reactant_mols.append(mol)
            
            product_mols = []
            for p_smiles in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smiles.strip())
                if mol:
                    product_mols.append(mol)
            
            if not reactant_mols or not product_mols:
                return False
            
            # Create SMARTS pattern for cyclopropane
            cyclopropane_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if not cyclopropane_pattern:
                return False
            
            # Count cyclopropane rings in reactants and products
            reactant_cyclopropane_count = sum(
                len(mol.GetSubstructMatches(cyclopropane_pattern)) 
                for mol in reactant_mols
            )
            
            product_cyclopropane_count = sum(
                len(mol.GetSubstructMatches(cyclopropane_pattern))
                for mol in product_mols
            )
            
            if self.direction == "formation":
                # Cyclopropane formation: more cyclopropanes in products than reactants
                return product_cyclopropane_count > reactant_cyclopropane_count
            else:  # "breaking"
                # Cyclopropane breaking: fewer cyclopropanes in products than reactants
                return reactant_cyclopropane_count > product_cyclopropane_count
                
        except Exception:
            return False
