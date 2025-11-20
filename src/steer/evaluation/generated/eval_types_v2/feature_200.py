"""Generated evaluation code for: Late stage Suzuki coupling for biaryl formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiCoupling(BaseScoring):
    """
    Evaluates whether a specific biaryl bond is formed via Suzuki coupling at a late stage.
    Checks for the formation of a phenyl-thiophene bond through Suzuki coupling reaction.
    """
    
    def __init__(self, config):
        self.bond_smarts = config["parameters"]["bond_smarts"]
        self.timing = config["parameters"]["timing"]
        self.reaction_type = config["parameters"]["reaction_type"]
        
        # Create pattern for the biaryl bond
        self.biaryl_pattern = Chem.MolFromSmarts(self.bond_smarts)
        
        # SMARTS patterns for Suzuki coupling components
        self.boronic_acid_pattern = Chem.MolFromSmarts("[c:1]-B([OH])[OH]")
        self.boronate_pattern = Chem.MolFromSmarts("[c:1]-B1OC(C)(C)C(C)(C)O1")
        self.halide_pattern = Chem.MolFromSmarts("[c:1][Br,I,Cl]")
        
    def route_scoring(self, x):
        """
        Convert depth fraction to score.
        For late-stage reactions, lower depth fractions (later in synthesis) get higher scores.
        """
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        else:
            if self.timing == "late":
                return (1 - x) * 10  # Late-stage gets higher score
            else:
                return x * 10  # Early-stage gets higher score
    
    def hit_condition(self, d):
        """
        Check if this reaction node represents a Suzuki coupling that forms the target biaryl bond.
        """
        metadata = d.get("metadata", {})
        
        # Check if we have reaction SMILES
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        # Parse molecules
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
        except:
            return False
        
        # Check if product contains the target biaryl pattern
        if not product.HasSubstructMatch(self.biaryl_pattern):
            return False
        
        # Check if this is a Suzuki coupling by looking for characteristic reactants
        has_boronic_component = False
        has_halide_component = False
        
        for reactant in reactants:
            if (reactant.HasSubstructMatch(self.boronic_acid_pattern) or 
                reactant.HasSubstructMatch(self.boronate_pattern)):
                has_boronic_component = True
            elif reactant.HasSubstructMatch(self.halide_pattern):
                has_halide_component = True
        
        # Suzuki coupling requires both boronic acid/ester and halide components
        if not (has_boronic_component and has_halide_component):
            return False
        
        # Verify that the biaryl bond is actually formed in this step
        # (i.e., it's not present in any individual reactant)
        biaryl_in_reactants = any(r.HasSubstructMatch(self.biaryl_pattern) for r in reactants)
        
        return not biaryl_in_reactants
