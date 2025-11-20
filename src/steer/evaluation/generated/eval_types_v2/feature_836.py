"""Generated evaluation code for: Late stage sulfamoylation of primary alcohol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSulfamoylation(BaseScoring):
    """
    Evaluates whether sulfamoylation of a primary alcohol occurs at late stage.
    Checks for the formation of sulfamate groups from primary alcohols within
    the specified depth threshold from the target molecule.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 2)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't happen
        else:
            # Late-stage is better, penalize if beyond threshold
            if x <= self.depth_threshold / 10.0:  # Convert to fraction
                return 1 - x  # Reward later stage reactions
            else:
                return 0.5 * (1 - x)  # Reduced reward for early reactions
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents sulfamoylation of primary alcohol"""
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles")
        if not rxn_smiles:
            return False
        
        try:
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # SMARTS pattern for sulfamate group (O-S(=O)(=O)-N)
            sulfamate_pattern = Chem.MolFromSmarts("[O;X2][S;X4](=[O;X1])(=[O;X1])[N;X3]")
            # SMARTS pattern for primary alcohol
            primary_alcohol_pattern = Chem.MolFromSmarts("[CH2;X4][OH;X2]")
            
            # Check if product contains sulfamate
            if not product.HasSubstructMatch(sulfamate_pattern):
                return False
            
            # Check if any reactant contains primary alcohol
            has_primary_alcohol_reactant = any(
                reactant.HasSubstructMatch(primary_alcohol_pattern) 
                for reactant in reactants
            )
            
            if not has_primary_alcohol_reactant:
                return False
            
            # Verify transformation: primary alcohol -> sulfamate
            # Get atom map numbers for the transformation
            product_atoms = {atom.GetAtomMapNum(): atom for atom in product.GetAtoms() if atom.GetAtomMapNum() > 0}
            
            # Find sulfamate oxygen in product
            sulfamate_matches = product.GetSubstructMatches(sulfamate_pattern)
            
            for match in sulfamate_matches:
                sulfamate_oxygen_idx = match[0]  # First atom in pattern is the oxygen
                sulfamate_oxygen = product.GetAtomByIdx(sulfamate_oxygen_idx)
                oxygen_map_num = sulfamate_oxygen.GetAtomMapNum()
                
                if oxygen_map_num == 0:
                    continue
                
                # Check if this oxygen was part of a primary alcohol in reactants
                for reactant in reactants:
                    reactant_atoms = {atom.GetAtomMapNum(): atom for atom in reactant.GetAtoms() if atom.GetAtomMapNum() > 0}
                    
                    if oxygen_map_num in reactant_atoms:
                        # Check if the mapped oxygen in reactant is part of primary alcohol
                        alcohol_matches = reactant.GetSubstructMatches(primary_alcohol_pattern)
                        for alcohol_match in alcohol_matches:
                            alcohol_oxygen_idx = alcohol_match[1]  # Second atom in pattern is OH oxygen
                            alcohol_oxygen = reactant.GetAtomByIdx(alcohol_oxygen_idx)
                            
                            if alcohol_oxygen.GetAtomMapNum() == oxygen_map_num:
                                return True
            
            return False
            
        except Exception:
            return False
