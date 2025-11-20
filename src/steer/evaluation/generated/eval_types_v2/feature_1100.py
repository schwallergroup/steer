"""Generated evaluation code for: Chiral auxiliary for stereocenter installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ChiralAuxiliaryUsage(BaseScoring):
    """
    Evaluates the use of chiral auxiliaries for stereocenter installation.
    Specifically looks for cyclic sulfamidate auxiliaries used for stereocontrol.
    """
    
    def __init__(self, config: Dict):
        self.auxiliary_type = config.get("auxiliary_type", "cyclic_sulfamidate")
        self.purpose = config.get("purpose", "stereocontrol")
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
        
        # Define SMARTS patterns for different chiral auxiliaries
        self.auxiliary_patterns = {
            "cyclic_sulfamidate": "[#7]1[#6][#6][#16](=[#8])(=[#8])[#8]1",  # Cyclic sulfamidate core
            "oxazolidinone": "[#7]1[#6](=[#8])[#8][#6][#6]1",  # Evans auxiliary
            "sultam": "[#7]1[#6][#6][#16](=[#8])(=[#8])1"  # Sultam auxiliary
        }
    
    def route_scoring(self, x) -> float:
        """Convert depth to score where earlier use of auxiliary is better."""
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0  # Auxiliary not used
            return 1 - x  # Earlier use is better (lower depth)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves a chiral auxiliary for stereocenter installation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for auxiliary introduction or utilization
            auxiliary_pattern = Chem.MolFromSmarts(self.auxiliary_patterns[self.auxiliary_type])
            if auxiliary_pattern is None:
                return False
            
            # Check if auxiliary is being introduced (appears in products but not reactants)
            auxiliary_in_reactants = any(mol.HasSubstructMatch(auxiliary_pattern) for mol in reactants)
            auxiliary_in_products = any(mol.HasSubstructMatch(auxiliary_pattern) for mol in products)
            
            # Auxiliary installation
            if auxiliary_in_products and not auxiliary_in_reactants:
                return True
            
            # Auxiliary-directed stereoselective reaction (auxiliary present in both)
            if auxiliary_in_reactants and auxiliary_in_products:
                # Check if a new stereocenter is being formed
                return self._check_stereocenter_formation(reactants, products)
            
            return False
            
        except Exception:
            return False
    
    def _check_stereocenter_formation(self, reactants, products):
        """
        Check if a new stereocenter is being formed in the presence of auxiliary.
        """
        try:
            # Count stereocenters in reactants vs products
            reactant_stereocenters = sum(len(Chem.FindMolChiralCenters(mol)) for mol in reactants)
            product_stereocenters = sum(len(Chem.FindMolChiralCenters(mol)) for mol in products)
            
            # New stereocenter formed
            if product_stereocenters > reactant_stereocenters:
                return True
            
            # Check for C-N bond formation near auxiliary (common pattern)
            for prod in products:
                for react in reactants:
                    if self._has_new_cn_bond_near_auxiliary(react, prod):
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _has_new_cn_bond_near_auxiliary(self, reactant, product):
        """
        Check if a new C-N bond is formed adjacent to the auxiliary structure.
        """
        try:
            auxiliary_pattern = Chem.MolFromSmarts(self.auxiliary_patterns[self.auxiliary_type])
            
            # Find auxiliary matches in both molecules
            react_matches = reactant.GetSubstructMatches(auxiliary_pattern)
            prod_matches = product.GetSubstructMatches(auxiliary_pattern)
            
            if not react_matches or not prod_matches:
                return False
            
            # Look for new C-N bonds adjacent to auxiliary
            cn_pattern = Chem.MolFromSmarts("[#6]-[#7]")
            react_cn_bonds = len(reactant.GetSubstructMatches(cn_pattern))
            prod_cn_bonds = len(product.GetSubstructMatches(cn_pattern))
            
            return prod_cn_bonds > react_cn_bonds
            
        except Exception:
            return False
