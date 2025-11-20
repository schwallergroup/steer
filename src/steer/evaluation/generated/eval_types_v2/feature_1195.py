"""Generated evaluation code for: Azide as amine protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AzideAmineProtectingGroup(BaseScoring):
    """
    Evaluates synthesis routes based on azide as amine protecting group strategy.
    Detects when an azide group is reduced to an amine, indicating the use of
    azide as a masked amine protecting group.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to 0-10 score"""
        if x < 0:
            return 0  # Strategy not used
        
        if self.condition_type == "bool":
            return 10  # Strategy is present
        else:
            # Earlier use of protecting group strategy is generally better
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents azide reduction to amine.
        Looks for azide (-N3) in reactants being converted to amine (-NH2) in products.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol:
                    reactant_mols.append(mol)
            
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol:
                    product_mols.append(mol)
            
            if not reactant_mols or not product_mols:
                return False
            
            # Define patterns for azide and primary amine
            azide_pattern = Chem.MolFromSmarts("[C,c]-N=[N+]=[N-]")  # Azide group
            primary_amine_pattern = Chem.MolFromSmarts("[C,c]-[NH2]")  # Primary amine
            
            if not azide_pattern or not primary_amine_pattern:
                return False
            
            # Check if any reactant has azide and any product has primary amine
            has_reactant_azide = any(mol.HasSubstructMatch(azide_pattern) for mol in reactant_mols)
            has_product_amine = any(mol.HasSubstructMatch(primary_amine_pattern) for mol in product_mols)
            
            # Additional check: ensure we don't have primary amine in reactants at the same position
            # This helps confirm it's a deprotection rather than just a coincidental reaction
            if has_reactant_azide and has_product_amine:
                # Get atom mappings to verify the transformation
                return self._verify_azide_to_amine_transformation(reactant_mols, product_mols)
            
            return False
            
        except Exception:
            return False
    
    def _verify_azide_to_amine_transformation(self, reactant_mols, product_mols):
        """
        Verify that the same carbon atom that was bonded to azide 
        is now bonded to amine in the products.
        """
        try:
            # Get all atom map numbers connected to azide in reactants
            azide_carbons = set()
            azide_pattern = Chem.MolFromSmarts("[C,c]-N=[N+]=[N-]")
            
            for mol in reactant_mols:
                matches = mol.GetSubstructMatches(azide_pattern)
                for match in matches:
                    carbon_idx = match[0]  # First atom in pattern is the carbon
                    carbon_atom = mol.GetAtomWithIdx(carbon_idx)
                    if carbon_atom.GetAtomMapNum() > 0:
                        azide_carbons.add(carbon_atom.GetAtomMapNum())
            
            # Get all atom map numbers connected to primary amine in products
            amine_carbons = set()
            amine_pattern = Chem.MolFromSmarts("[C,c]-[NH2]")
            
            for mol in product_mols:
                matches = mol.GetSubstructMatches(amine_pattern)
                for match in matches:
                    carbon_idx = match[0]  # First atom in pattern is the carbon
                    carbon_atom = mol.GetAtomWithIdx(carbon_idx)
                    if carbon_atom.GetAtomMapNum() > 0:
                        amine_carbons.add(carbon_atom.GetAtomMapNum())
            
            # Check if there's overlap - same carbon went from azide to amine
            return len(azide_carbons.intersection(amine_carbons)) > 0
            
        except Exception:
            # If mapping analysis fails, fall back to simpler check
            return True
