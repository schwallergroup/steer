"""Generated evaluation code for: Late stage amide coupling to cephalosporin core"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Evaluates whether amide coupling to a cephalosporin core occurs at a late stage.
    Checks for the presence of amide bond formation involving a cephalosporin substrate
    and rewards reactions that happen closer to the final product.
    """
    
    def __init__(self, config: Dict):
        self.substrate_pattern = config["parameters"]["substrate_pattern"]
        self.timing = config["parameters"]["timing"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Amide coupling doesn't happen
        else:
            # Late-stage coupling is better (lower depth fraction is rewarded)
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves amide coupling to cephalosporin core
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants_smiles = rxn[0].split(".")
            product_smiles = rxn[1]
            
            # Parse molecules
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles if smi]
            product = Chem.MolFromSmiles(product_smiles)
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains cephalosporin core
            cephalosporin_pattern = Chem.MolFromSmarts(self.substrate_pattern)
            if not product.HasSubstructMatch(cephalosporin_pattern):
                return False
            
            # Check if one reactant contains cephalosporin core (substrate)
            has_cephalosporin_reactant = any(
                mol.HasSubstructMatch(cephalosporin_pattern) for mol in reactants
            )
            
            if not has_cephalosporin_reactant:
                return False
            
            # Check for amide bond formation pattern
            # Look for C(=O)N pattern in product that's not in reactants
            amide_pattern = Chem.MolFromSmarts("[C](=O)[N]")
            product_amides = len(product.GetSubstructMatches(amide_pattern))
            reactant_amides = sum(len(mol.GetSubstructMatches(amide_pattern)) for mol in reactants)
            
            # Check if new amide bonds are formed
            if product_amides <= reactant_amides:
                return False
            
            # Additional check: look for acyl chloride or carboxylic acid reactant
            # which would indicate amide coupling
            acyl_chloride_pattern = Chem.MolFromSmarts("[C](=O)[Cl]")
            carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")
            
            has_acylating_agent = any(
                mol.HasSubstructMatch(acyl_chloride_pattern) or 
                mol.HasSubstructMatch(carboxylic_acid_pattern)
                for mol in reactants
            )
            
            return has_acylating_agent
            
        except Exception:
            return False
