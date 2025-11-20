"""Generated evaluation code for: Late stage aldehyde to carboxylic acid oxidation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAldehydeOxidation(BaseScoring):
    """
    Evaluates routes for late-stage aldehyde to carboxylic acid oxidation.
    Checks if an aldehyde functional group is oxidized to a carboxylic acid
    in the later stages of the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
        
        # SMARTS patterns for functional groups
        self.aldehyde_pattern = Chem.MolFromSmarts("[CX3H1](=O)")
        self.carboxylic_acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Oxidation doesn't happen
        else:
            # Later oxidation is better (higher x values preferred)
            if self.condition_type == "bool":
                return 10 if x >= self.target_depth else 0
            else:
                # Reward late-stage oxidation
                return max(0, 10 * (x - 0.5) / 0.5) if x > 0.5 else 0

    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step performs aldehyde to carboxylic acid oxidation.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            product_smiles, reactant_smiles = rxn_smiles.split(">>")
            product_mol = Chem.MolFromSmiles(product_smiles)
            
            if product_mol is None:
                return False
            
            # Check if product contains carboxylic acid
            if not product_mol.HasSubstructMatch(self.carboxylic_acid_pattern):
                return False
            
            # Get all reactant molecules
            reactant_mols = []
            for r_smiles in reactant_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles)
                if mol is not None:
                    reactant_mols.append(mol)
            
            if not reactant_mols:
                return False
            
            # Check if any reactant contains aldehyde
            has_aldehyde_reactant = any(
                mol.HasSubstructMatch(self.aldehyde_pattern) for mol in reactant_mols
            )
            
            if not has_aldehyde_reactant:
                return False
            
            # Verify oxidation by checking atom mapping
            # Find carboxylic acid carbons in product
            product_matches = product_mol.GetSubstructMatches(self.carboxylic_acid_pattern)
            
            for match in product_matches:
                carbox_carbon_idx = match[0]  # First atom in SMARTS is the carbon
                carbox_carbon = product_mol.GetAtomWithIdx(carbox_carbon_idx)
                carbox_map_num = carbox_carbon.GetAtomMapNum()
                
                if carbox_map_num == 0:
                    continue
                
                # Find corresponding carbon in reactants
                for reactant_mol in reactant_mols:
                    for atom in reactant_mol.GetAtoms():
                        if atom.GetAtomMapNum() == carbox_map_num:
                            # Check if this atom is part of an aldehyde
                            for aldehyde_match in reactant_mol.GetSubstructMatches(self.aldehyde_pattern):
                                if atom.GetIdx() in aldehyde_match:
                                    return True
            
            return False
            
        except Exception:
            return False
