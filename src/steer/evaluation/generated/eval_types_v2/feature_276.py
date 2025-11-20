"""Generated evaluation code for: Azide displacement for stereocontrolled amine introduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AzideDisplacement(BaseScoring):
    """
    Evaluates synthesis routes for azide displacement reactions that provide
    stereocontrolled amine introduction via SN2 displacement of mesylate/tosylate
    with azide, offering stereochemical inversion and reliable amine precursor.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.0)
        
        # Define SMARTS pattern for azide displacement
        self.azide_pattern = "[C:1][S:2](=O)(=O)[O:3].[N-:4]=[N+:5]=[N-:6]>>[C:1][N:4]=[N+:5]=[N-:6]"
        
        # Alternative patterns for mesylate/tosylate displacement
        self.mesylate_pattern = Chem.MolFromSmarts("[C][S](=O)(=O)[O]")
        self.tosylate_pattern = Chem.MolFromSmarts("[C][S](=O)(=O)[O][c]1[c][c][c]([C])[c][c]1")
        self.azide_nucleophile = Chem.MolFromSmarts("[N-]=[N+]=[N-]")
        self.azide_product = Chem.MolFromSmarts("[C][N]=[N+]=[N-]")

    def route_scoring(self, x) -> float:
        """Convert depth fraction to 0-10 score."""
        if x < 0:
            return 0  # Reaction not found
        
        if self.condition_type == "bool":
            return 10  # Found the reaction
        else:
            # Earlier in synthesis is better for building block preparation
            return 10 * (1 - x)

    def hit_condition(self, d) -> bool:
        """Check if a reaction node represents azide displacement."""
        try:
            metadata = d.get("metadata", {})
            mapped_rxn = metadata.get("mapped_reaction_smiles", "")
            
            if not mapped_rxn or ">>" not in mapped_rxn:
                return False
            
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse reactants and products
            reactant_mols = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi.strip())
                if mol:
                    reactant_mols.append(mol)
            
            product_mols = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi.strip())
                if mol:
                    product_mols.append(mol)
            
            if not reactant_mols or not product_mols:
                return False
            
            # Check for azide nucleophile in reactants
            has_azide_nucleophile = any(
                mol.HasSubstructMatch(self.azide_nucleophile) 
                for mol in reactant_mols
            )
            
            # Check for leaving group (mesylate/tosylate) in reactants
            has_leaving_group = any(
                mol.HasSubstructMatch(self.mesylate_pattern) or 
                mol.HasSubstructMatch(self.tosylate_pattern)
                for mol in reactant_mols
            )
            
            # Check for azide product formation
            has_azide_product = any(
                mol.HasSubstructMatch(self.azide_product)
                for mol in product_mols
            )
            
            # All conditions must be met for azide displacement
            return has_azide_nucleophile and has_leaving_group and has_azide_product
            
        except Exception:
            return False
