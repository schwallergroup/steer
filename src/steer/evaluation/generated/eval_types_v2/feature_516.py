"""Generated evaluation code for: Mitsunobu inversion for stereochemical control"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MitsunobuInversion(BaseScoring):
    """
    Evaluates if a Mitsunobu reaction is used for stereochemical inversion,
    specifically converting secondary alcohol to azide with inversion of configuration.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", 0)
        
        # SMARTS patterns for detection
        self.secondary_alcohol_pattern = "[CX4H1]([OH1])"  # Secondary alcohol
        self.azide_pattern = "[NX2]=[NX2]=[NX1]"  # Azide group
        self.dppa_pattern = "[P](=[O])([N-][N+]#N)"  # DPPA reagent pattern
        self.triphenylphosphine_pattern = "P(c1ccccc1)(c2ccccc2)(c3ccccc3)"  # PPh3
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to 0-10 score."""
        if self.condition_type == "bool":
            if x < 0:
                return 0  # Condition not met
            else:
                return 10  # Condition met
        else:
            if x < 0:
                return 0
            # Earlier Mitsunobu reactions are generally better for stereochemical control
            return max(0, 10 - abs(x - self.target_depth) * 5)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents a Mitsunobu inversion."""
        try:
            metadata = d.get("metadata", {})
            mapped_rxn = metadata.get("mapped_reaction_smiles", "")
            
            if not mapped_rxn or ">>" not in mapped_rxn:
                return False
            
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for presence of secondary alcohol in reactants
            has_sec_alcohol = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.secondary_alcohol_pattern))
                for mol in reactants
            )
            
            # Check for azide formation in products
            has_azide = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.azide_pattern))
                for mol in products
            )
            
            # Check for Mitsunobu reagents (DPPA or PPh3/DEAD combination)
            has_mitsunobu_reagent = (
                any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.dppa_pattern)) for mol in reactants) or
                any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.triphenylphosphine_pattern)) for mol in reactants)
            )
            
            # Additional check for policy name if available
            policy_check = metadata.get("policy_name", "").lower()
            has_mitsunobu_policy = "mitsunobu" in policy_check or "inversion" in policy_check
            
            # Confirm alcohol-to-azide transformation with Mitsunobu conditions
            return (has_sec_alcohol and has_azide and 
                   (has_mitsunobu_reagent or has_mitsunobu_policy))
            
        except Exception:
            return False
