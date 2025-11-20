"""Generated evaluation code for: Early acid chloride formation strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyAcidChlorideFormation(BaseScoring):
    """
    Evaluates whether acid chloride formation occurs early in the synthesis route.
    Rewards routes where acid chloride (COCl) functional group is formed in early steps
    and potentially carried through subsequent transformations.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "early")
        self.functional_group = config.get("functional_group", "COCl")
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10 scale).
        Early formation (low x) gets higher score.
        """
        if x < 0:
            return 0  # Acid chloride formation doesn't occur
        
        if self.timing_preference == "early":
            # Reward early formation: score decreases as depth increases
            return max(0, 10 * (1 - x))
        else:
            # For other timing preferences, use simpler scoring
            return 10 if x >= 0 else 0
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node involves acid chloride formation.
        Detects formation of COCl group from carboxylic acid or derivative.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol:
                    reactant_mols.append(mol)
            
            product_mol = Chem.MolFromSmiles(products_smiles.strip())
            
            if not product_mol or not reactant_mols:
                return False
            
            # Check if product contains acid chloride group
            acid_chloride_pattern = Chem.MolFromSmarts("[C](=[O])[Cl]")
            if not product_mol.HasSubstructMatch(acid_chloride_pattern):
                return False
            
            # Check if reactants contain carboxylic acid or acid derivative
            carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=[O])[OH]")
            acid_derivative_patterns = [
                Chem.MolFromSmarts("[C](=[O])[O][C]"),  # Ester
                Chem.MolFromSmarts("[C](=[O])[N]"),     # Amide
                Chem.MolFromSmarts("[C](=[O])[O][C](=[O])") # Anhydride
            ]
            
            # Check if any reactant has carboxylic acid or derivative
            has_acid_precursor = False
            for mol in reactant_mols:
                if mol.HasSubstructMatch(carboxylic_acid_pattern):
                    has_acid_precursor = True
                    break
                for pattern in acid_derivative_patterns:
                    if mol.HasSubstructMatch(pattern):
                        has_acid_precursor = True
                        break
                if has_acid_precursor:
                    break
            
            # Also check for presence of chlorinating agents
            chlorinating_agents = [
                Chem.MolFromSmarts("[S](=[O])(=[O])[Cl]"),  # SOCl2
                Chem.MolFromSmarts("[P][Cl]"),              # PCl3, PCl5
                Chem.MolFromSmarts("C(=O)Cl"),              # Phosgene-like
            ]
            
            has_chlorinating_agent = False
            for mol in reactant_mols:
                for agent_pattern in chlorinating_agents:
                    if mol.HasSubstructMatch(agent_pattern):
                        has_chlorinating_agent = True
                        break
                if has_chlorinating_agent:
                    break
            
            # Return True if we have acid precursor and evidence of chlorination
            return has_acid_precursor and (has_chlorinating_agent or self._check_chloride_increase(reactant_mols, product_mol))
            
        except Exception:
            return False
    
    def _check_chloride_increase(self, reactant_mols, product_mol) -> bool:
        """
        Helper method to check if chlorine atoms increased from reactants to product,
        indicating potential acid chloride formation.
        """
        try:
            reactant_cl_count = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[Cl]"))) 
                                  for mol in reactant_mols)
            product_cl_count = len(product_mol.GetSubstructMatches(Chem.MolFromSmarts("[Cl]")))
            
            # Check if chlorine was incorporated into carbonyl context
            product_cocl_count = len(product_mol.GetSubstructMatches(Chem.MolFromSmarts("[C](=[O])[Cl]")))
            
            return product_cocl_count > 0 and product_cl_count >= reactant_cl_count
        except Exception:
            return False
