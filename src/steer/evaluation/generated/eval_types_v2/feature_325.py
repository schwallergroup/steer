"""Generated evaluation code for: Early Baeyer-Villiger oxidation to install phenol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BaeyerVilligerEarly(BaseScoring):
    """
    Evaluates whether a Baeyer-Villiger oxidation occurs early in the synthesis route.
    Detects the conversion of aryl methyl ketones to phenolic acetates.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)  # Early = top 20% of route
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10). Early reactions get higher scores."""
        if x < 0:
            return 0  # Reaction not found
        
        if self.condition_type == "bool":
            return 10 if x <= self.target_depth else 0
        else:
            # Score inversely related to depth - earlier is better
            if x <= self.target_depth:
                return 10
            else:
                # Gradual decrease for later occurrences
                return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """Check if a reaction node represents a Baeyer-Villiger oxidation."""
        metadata = d.get("metadata", {})
        
        # Check if mapped reaction SMILES is available
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if any(mol is None for mol in reactant_mols + product_mols):
                return False
            
            return self._detect_baeyer_villiger(reactant_mols, product_mols)
            
        except Exception:
            return False
    
    def _detect_baeyer_villiger(self, reactants, products) -> bool:
        """
        Detect Baeyer-Villiger oxidation pattern:
        Aryl methyl ketone -> phenolic acetate (or similar ester)
        """
        # Pattern for aryl methyl ketone (aromatic ring connected to C(=O)CH3)
        aryl_methyl_ketone_pattern = Chem.MolFromSmarts("[cH1,c]1[cH1,c][cH1,c][cH1,c][cH1,c][cH1,c]1-C(=O)-[CH3]")
        
        # Pattern for phenolic acetate/ester (aromatic OH with nearby acetate)
        phenolic_ester_pattern = Chem.MolFromSmarts("[cH1,c]1[cH1,c][cH1,c][cH1,c]([OH1])[cH1,c][cH1,c]1")
        ester_pattern = Chem.MolFromSmarts("C(=O)-O-[cH1,c]1[cH1,c][cH1,c][cH1,c][cH1,c][cH1,c]1")
        
        # Check for aryl methyl ketone in reactants
        has_aryl_ketone = any(
            mol.HasSubstructMatch(aryl_methyl_ketone_pattern) 
            for mol in reactants
        )
        
        if not has_aryl_ketone:
            return False
        
        # Check for phenolic ester formation in products
        has_phenolic_ester = any(
            mol.HasSubstructMatch(phenolic_ester_pattern) and mol.HasSubstructMatch(ester_pattern)
            for mol in products
        )
        
        # Alternative: check for general ester formation from ketone
        if not has_phenolic_ester:
            # Look for ester pattern in products that wasn't in reactants
            product_has_ester = any(
                mol.HasSubstructMatch(ester_pattern) 
                for mol in products
            )
            reactant_has_ester = any(
                mol.HasSubstructMatch(ester_pattern) 
                for mol in reactants
            )
            
            has_phenolic_ester = product_has_ester and not reactant_has_ester
        
        return has_aryl_ketone and has_phenolic_ester
