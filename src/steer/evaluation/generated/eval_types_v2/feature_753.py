"""Generated evaluation code for: Late stage amide coupling for fragment assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Evaluates whether amide coupling reactions occur within a specified late-stage depth range.
    
    Detects amide bond formation reactions and scores routes based on whether the coupling
    occurs at the desired late stage (shallow depth) for fragment assembly.
    """
    
    def __init__(self, config: Dict):
        self.min_depth = config["depth_range"][0]
        self.max_depth = config["depth_range"][1]
        self.timing = config.get("timing", "late")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Amide coupling doesn't happen
        
        # For late-stage preference, earlier (lower depth) is better
        if self.timing == "late":
            if x <= self.min_depth:
                return 1.0  # Perfect late-stage timing
            elif x <= self.max_depth:
                # Linear decay from 1.0 to 0.3 within acceptable range
                return 1.0 - 0.7 * (x - self.min_depth) / (self.max_depth - self.min_depth)
            else:
                return 0.1  # Too early in synthesis
        else:
            # Standard depth scoring - closer to target range is better
            if self.min_depth <= x <= self.max_depth:
                return 1.0
            else:
                # Distance penalty from nearest range boundary
                if x < self.min_depth:
                    penalty = (self.min_depth - x) * 0.2
                else:
                    penalty = (x - self.max_depth) * 0.2
                return max(0.1, 1.0 - penalty)
    
    def hit_condition(self, d) -> bool:
        """
        Detect amide coupling reactions by identifying amide bond formation patterns.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, product_smiles = mapped_rxn.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Define amide patterns to detect in product
            amide_patterns = [
                Chem.MolFromSmarts("[C:1](=[O:2])[NH:3]"),  # Primary amide
                Chem.MolFromSmarts("[C:1](=[O:2])[NH1:3][C:4]"),  # Secondary amide
                Chem.MolFromSmarts("[C:1](=[O:2])[N:3]([C:4])[C:5]")  # Tertiary amide
            ]
            
            # Find amide bonds in product
            product_amide_atoms = set()
            for pattern in amide_patterns:
                if pattern:
                    matches = product.GetSubstructMatches(pattern)
                    for match in matches:
                        # Store the C=O and N atoms involved in amide bond
                        product_amide_atoms.update([match[0], match[2]])  # C and N atoms
            
            if not product_amide_atoms:
                return False
            
            # Check if this amide bond is newly formed (not present in reactants)
            for reactant in reactants:
                reactant_amide_atoms = set()
                for pattern in amide_patterns:
                    if pattern:
                        matches = reactant.GetSubstructMatches(pattern)
                        for match in matches:
                            # Get atom map numbers for C and N in amide
                            c_atom = reactant.GetAtomWithIdx(match[0])
                            n_atom = reactant.GetAtomWithIdx(match[2])
                            c_mapnum = c_atom.GetAtomMapNum()
                            n_mapnum = n_atom.GetAtomMapNum()
                            if c_mapnum > 0 and n_mapnum > 0:
                                reactant_amide_atoms.update([c_mapnum, n_mapnum])
                
                # Remove any pre-existing amide bonds from consideration
                product_mapped_amides = set()
                for pattern in amide_patterns:
                    if pattern:
                        matches = product.GetSubstructMatches(pattern)
                        for match in matches:
                            c_atom = product.GetAtomWithIdx(match[0])
                            n_atom = product.GetAtomWithIdx(match[2])
                            c_mapnum = c_atom.GetAtomMapNum()
                            n_mapnum = n_atom.GetAtomMapNum()
                            if c_mapnum > 0 and n_mapnum > 0:
                                if not ({c_mapnum, n_mapnum} & reactant_amide_atoms):
                                    product_mapped_amides.update([c_mapnum, n_mapnum])
                
                if product_mapped_amides:
                    return True
            
            # Additional check: look for typical amide coupling reactant patterns
            coupling_patterns = [
                Chem.MolFromSmarts("[C:1](=[O:2])[OH]"),  # Carboxylic acid
                Chem.MolFromSmarts("[C:1](=[O:2])[Cl]"),   # Acid chloride  
                Chem.MolFromSmarts("[NH2:1]"),             # Primary amine
                Chem.MolFromSmarts("[NH1:1]"),             # Secondary amine
            ]
            
            has_acid_component = False
            has_amine_component = False
            
            for reactant in reactants:
                # Check for acid/acyl component
                if any(reactant.HasSubstructMatch(pattern) for pattern in coupling_patterns[:2]):
                    has_acid_component = True
                # Check for amine component  
                if any(reactant.HasSubstructMatch(pattern) for pattern in coupling_patterns[2:]):
                    has_amine_component = True
            
            return has_acid_component and has_amine_component
            
        except Exception:
            return False
