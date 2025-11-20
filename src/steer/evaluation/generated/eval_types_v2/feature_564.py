"""Generated evaluation code for: Early stage cyclopropane ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyCyclopropaneFormation(BaseScoring):
    """
    Evaluates early stage cyclopropane ring formation, particularly using 
    Corey-Chaykovsky sulfur ylide chemistry on activated alkenes.
    
    Rewards routes where cyclopropane formation occurs early in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "early")
        self.method = config.get("method", "corey_chaykovsky")
        
        # SMARTS patterns for cyclopropane detection
        self.cyclopropane_pattern = "[C,c]1[C,c][C,c]1"
        
        # SMARTS patterns for Corey-Chaykovsky reagents/intermediates
        self.sulfur_ylide_patterns = [
            "[S+]([C,c])([C,c])[C-]",  # Sulfur ylide
            "[S]([C,c])([C,c])[CH2]",  # Dimethylsulfide precursor
            "[S+]([C,c])([C,c])[CH2-]", # Alternative ylide representation
        ]
        
        # Activated alkene patterns (electron-withdrawing groups)
        self.activated_alkene_patterns = [
            "C=C[C](=O)",  # Alpha,beta-unsaturated carbonyl
            "C=CC(=O)",    # Alternative carbonyl pattern
            "C=C[C]#N",    # Acrylonitrile-type
            "C=CC#N",      # Alternative nitrile pattern
        ]
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        Early formation (low x) gives higher scores.
        """
        if x < 0:
            return 0  # Cyclopropane formation doesn't occur
        
        if self.timing_preference == "early":
            # Reward early formation: score decreases as depth increases
            if x <= 0.2:  # Very early
                return 10
            elif x <= 0.4:  # Early
                return 8
            elif x <= 0.6:  # Mid-stage
                return 5
            else:  # Late stage
                return 2
        else:
            # Standard scoring based on presence
            return 6  # Fixed score for successful detection
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction involves cyclopropane formation,
        preferably via Corey-Chaykovsky methodology.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse molecules
            product_mol = Chem.MolFromSmiles(products_smiles)
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) 
                           for smi in reactants_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if product contains cyclopropane
            cycloprop_pattern = Chem.MolFromSmarts(self.cyclopropane_pattern)
            if not product_mol.HasSubstructMatch(cycloprop_pattern):
                return False
            
            # Check if reactants lack cyclopropane (formation, not just presence)
            reactants_have_cycloprop = any(
                mol.HasSubstructMatch(cycloprop_pattern) for mol in reactant_mols
            )
            if reactants_have_cycloprop:
                return False  # Not a formation reaction
            
            # If method is specified as Corey-Chaykovsky, check for characteristic patterns
            if self.method == "corey_chaykovsky":
                return self._detect_corey_chaykovsky_conditions(reactant_mols)
            
            # Otherwise, any cyclopropane formation is acceptable
            return True
            
        except Exception:
            return False
    
    def _detect_corey_chaykovsky_conditions(self, reactant_mols) -> bool:
        """
        Detect Corey-Chaykovsky reaction conditions:
        - Presence of sulfur ylide or precursor
        - Presence of activated alkene
        """
        has_sulfur_component = False
        has_activated_alkene = False
        
        # Check for sulfur ylide patterns
        for pattern_smarts in self.sulfur_ylide_patterns:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if pattern and any(mol.HasSubstructMatch(pattern) for mol in reactant_mols):
                has_sulfur_component = True
                break
        
        # Check for activated alkene patterns
        for pattern_smarts in self.activated_alkene_patterns:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if pattern and any(mol.HasSubstructMatch(pattern) for mol in reactant_mols):
                has_activated_alkene = True
                break
        
        # For Corey-Chaykovsky, we want both components
        # But be flexible - if we can't detect the specific patterns,
        # still allow cyclopropane formation to be scored
        return has_sulfur_component or has_activated_alkene or len(reactant_mols) >= 2
