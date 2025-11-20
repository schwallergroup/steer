"""Generated evaluation code for: Late stage cyclopropanation on complex substrate"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSpecificReaction(BaseScoring):
    """
    Evaluates whether a specific reaction type occurs at a target position from the end of the route.
    Designed for late-stage cyclopropanation but generalizable to other reaction types.
    """
    
    def __init__(self, config: Dict):
        self.reaction_type = config["parameters"]["reaction_type"]
        self.step_position_from_end = config["parameters"]["step_position_from_end"]
        
        # Define reaction patterns for different types
        self.reaction_patterns = {
            "cyclopropanation": {
                "reactant_pattern": "C=C",  # Alkene substrate
                "product_pattern": "C1CC1",  # Cyclopropane ring
                "reagent_patterns": ["C(=O)C(Br)Br", "C(=O)CHI2", "N2C=CC=C2"]  # Common cyclopropanation reagents
            }
        }
    
    def route_scoring(self, x) -> float:
        """
        Score based on whether reaction occurs at target position.
        x is the depth fraction where the reaction occurs (-1 if not found).
        """
        if x < 0:
            return 0  # Reaction type not found
        
        # Calculate actual step position from end based on depth fraction
        # x=1.0 means last step (position 1 from end)
        # x=0.5 means middle, etc.
        actual_position_from_end = x
        target_position_normalized = self.step_position_from_end / 10.0  # Normalize to 0-1 range
        
        # Perfect score if at exact target position, decreasing with distance
        position_diff = abs(actual_position_from_end - target_position_normalized)
        
        if position_diff < 0.1:  # Within 10% of target position
            return 10.0
        else:
            # Linear decay - further from target position = lower score
            return max(0, 10.0 - (position_diff * 50))
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction node represents the target reaction type.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    reactant_mols.append(mol)
            
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    product_mols.append(mol)
            
            if not reactant_mols or not product_mols:
                return False
            
            return self._detect_reaction_type(reactant_mols, product_mols)
            
        except Exception:
            return False
    
    def _detect_reaction_type(self, reactants, products) -> bool:
        """
        Detect if the reaction matches the specified reaction type.
        """
        if self.reaction_type == "cyclopropanation":
            return self._detect_cyclopropanation(reactants, products)
        
        return False
    
    def _detect_cyclopropanation(self, reactants, products) -> bool:
        """
        Detect cyclopropanation by checking for:
        1. Alkene in reactants that becomes cyclopropane in products
        2. Presence of typical cyclopropanation reagents
        """
        patterns = self.reaction_patterns["cyclopropanation"]
        
        # Check for alkene substrate in reactants
        alkene_pattern = Chem.MolFromSmarts(patterns["reactant_pattern"])
        has_alkene_reactant = any(mol.HasSubstructMatch(alkene_pattern) for mol in reactants)
        
        if not has_alkene_reactant:
            return False
        
        # Check for cyclopropane formation in products
        cyclopropane_pattern = Chem.MolFromSmarts(patterns["product_pattern"])
        has_cyclopropane_product = any(mol.HasSubstructMatch(cyclopropane_pattern) for mol in products)
        
        if not has_cyclopropane_product:
            return False
        
        # Additional check: look for typical cyclopropanation reagents
        reagent_found = False
        for reagent_smarts in patterns["reagent_patterns"]:
            reagent_pattern = Chem.MolFromSmarts(reagent_smarts)
            if any(mol.HasSubstructMatch(reagent_pattern) for mol in reactants):
                reagent_found = True
                break
        
        # Also check for carbene precursors or diazo compounds
        diazo_pattern = Chem.MolFromSmarts("C=[N+]=[N-]")  # Diazo group
        simmons_smith_pattern = Chem.MolFromSmarts("[CH2][Zn]")  # Simmons-Smith reagent
        
        carbene_reagent = any(mol.HasSubstructMatch(diazo_pattern) or 
                            mol.HasSubstructMatch(simmons_smith_pattern) 
                            for mol in reactants)
        
        return has_alkene_reactant and has_cyclopropane_product and (reagent_found or carbene_reagent)
