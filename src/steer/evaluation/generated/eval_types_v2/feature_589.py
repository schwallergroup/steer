"""Generated evaluation code for: Late stage Sonogashira coupling for aryl-alkyne bond"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SonogashiraCoupling(BaseScoring):
    """
    Evaluates synthesis routes based on when a Sonogashira coupling reaction occurs
    to form an aryl-alkyne bond. Rewards late-stage formation of the specified bond.
    """
    
    def __init__(self, config: Dict):
        self.bond_smarts = config["parameters"]["bond_smarts"]
        self.reaction_type = config["parameters"]["reaction_type"]
        self.timing = config["parameters"]["timing"]
        
        # Compile the SMARTS pattern for the aryl-alkyne bond
        self.bond_pattern = Chem.MolFromSmarts(self.bond_smarts)
        
        # Sonogashira reaction patterns (aryl halide + terminal alkyne -> aryl-alkyne)
        self.sonogashira_patterns = [
            "[c:1][X].[C:2]#[C:3][H]>>[c:1][C:2]#[C:3]",  # Basic pattern
            "[c:1][Br,I,Cl].[C:2]#[C:3][H]>>[c:1][C:2]#[C:3]"  # With specific halogens
        ]

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Sonogashira coupling doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Later stage is better (higher score for smaller depth fraction)
            elif self.timing == "early":
                return x  # Earlier stage is better
            else:
                return 1  # Just presence is rewarded

    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents a Sonogashira coupling that forms
        the target aryl-alkyne bond.
        """
        try:
            # Get the mapped reaction SMILES
            mapped_rxn = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = mapped_rxn.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains the target aryl-alkyne bond
            if not product.HasSubstructMatch(self.bond_pattern):
                return False
            
            # Check if this is a Sonogashira-type reaction
            if not self._is_sonogashira_reaction(product, reactants):
                return False
            
            # Check if the specific bond was formed (not already present in reactants)
            return self._bond_was_formed(product, reactants)
            
        except Exception:
            return False

    def _is_sonogashira_reaction(self, product, reactants) -> bool:
        """Check if this reaction matches Sonogashira coupling patterns."""
        # Look for aryl halide + terminal alkyne pattern
        has_aryl_halide = False
        has_terminal_alkyne = False
        
        aryl_halide_pattern = Chem.MolFromSmarts("[c][Br,I,Cl]")
        terminal_alkyne_pattern = Chem.MolFromSmarts("[C]#[C][H]")
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(aryl_halide_pattern):
                has_aryl_halide = True
            if reactant.HasSubstructMatch(terminal_alkyne_pattern):
                has_terminal_alkyne = True
        
        return has_aryl_halide and has_terminal_alkyne

    def _bond_was_formed(self, product, reactants) -> bool:
        """
        Check if the target aryl-alkyne bond was actually formed in this reaction
        (i.e., it's present in product but not in any single reactant).
        """
        # Get atom map numbers for the bond atoms in the product
        matches = product.GetSubstructMatches(self.bond_pattern)
        
        if not matches:
            return False
        
        for match in matches:
            aryl_carbon_map = None
            alkyne_carbon_map = None
            
            # Get atom map numbers for the bonded atoms
            for atom_idx in match:
                atom = product.GetAtomByIdx(atom_idx)
                map_num = atom.GetAtomMapNum()
                if map_num > 0:
                    if atom.GetIsAromatic():
                        aryl_carbon_map = map_num
                    elif atom.GetHybridization() == Chem.HybridizationType.SP:
                        alkyne_carbon_map = map_num
            
            if aryl_carbon_map and alkyne_carbon_map:
                # Check if these atoms were in different reactants
                aryl_reactant = None
                alkyne_reactant = None
                
                for reactant in reactants:
                    reactant_maps = [a.GetAtomMapNum() for a in reactant.GetAtoms()]
                    if aryl_carbon_map in reactant_maps:
                        aryl_reactant = reactant
                    if alkyne_carbon_map in reactant_maps:
                        alkyne_reactant = reactant
                
                # Bond was formed if the atoms were in different reactants
                if aryl_reactant != alkyne_reactant and aryl_reactant and alkyne_reactant:
                    return True
        
        return False
