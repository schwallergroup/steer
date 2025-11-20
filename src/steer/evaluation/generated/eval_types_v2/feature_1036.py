"""Generated evaluation code for: Convergent synthesis via Suzuki coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSuzuki(BaseScoring):
    """
    Evaluates convergent synthesis routes that use Suzuki-Miyaura coupling.
    Checks for the presence of Suzuki coupling reaction and validates that
    it occurs at an appropriate stage with the expected number of fragments.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["parameters"].get("fragment_count", 2)
        self.target_stage = config["parameters"].get("stage", "mid")
        self.coupling_reaction = config["parameters"].get("coupling_reaction", "Suzuki-Miyaura")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling not found
        
        # Score based on when coupling occurs relative to target stage
        if self.target_stage == "early":
            target_fraction = 0.2
        elif self.target_stage == "mid":
            target_fraction = 0.5
        else:  # late
            target_fraction = 0.8
            
        # Higher score for coupling occurring closer to target stage
        deviation = abs(x - target_fraction)
        return max(0, 1 - deviation * 2)  # Scale to 0-1 range
    
    def hit_condition(self, d):
        """Check if this reaction is a Suzuki-Miyaura coupling with correct fragment count"""
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        # Check if reaction involves Suzuki coupling pattern
        if not self._is_suzuki_coupling(rxn_smiles):
            return False
            
        # Verify fragment count in reactants
        reactants_smiles = rxn_smiles.split(">>")[1]
        reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
        reactant_mols = [mol for mol in reactant_mols if mol is not None]
        
        # Count significant fragments (exclude small molecules like bases, catalysts)
        significant_fragments = self._count_significant_fragments(reactant_mols)
        
        return significant_fragments == self.fragment_count
    
    def _is_suzuki_coupling(self, rxn_smiles):
        """Detect Suzuki-Miyaura coupling by checking for boronic acid/ester and aryl halide"""
        reactants_smiles = rxn_smiles.split(">>")[1]
        reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
        reactant_mols = [mol for mol in reactant_mols if mol is not None]
        
        # Patterns for Suzuki coupling components
        boronic_acid_pattern = Chem.MolFromSmarts("[#6]-B(-O)-O")  # Boronic acid
        boronic_ester_pattern = Chem.MolFromSmarts("[#6]-B1-O-C-C-O-1")  # Boronic ester (pinacol)
        aryl_halide_pattern = Chem.MolFromSmarts("c-[Cl,Br,I]")  # Aryl halide
        triflate_pattern = Chem.MolFromSmarts("c-O-S(=O)(=O)-C(F)(F)F")  # Triflate
        
        has_boron_component = False
        has_electrophile = False
        
        for mol in reactant_mols:
            # Check for boronic acid or ester
            if (mol.HasSubstructMatch(boronic_acid_pattern) or 
                mol.HasSubstructMatch(boronic_ester_pattern)):
                has_boron_component = True
                
            # Check for aryl halide or triflate
            if (mol.HasSubstructMatch(aryl_halide_pattern) or 
                mol.HasSubstructMatch(triflate_pattern)):
                has_electrophile = True
                
        return has_boron_component and has_electrophile
    
    def _count_significant_fragments(self, reactant_mols):
        """Count fragments that are significant (not small catalysts/bases)"""
        significant_count = 0
        
        for mol in reactant_mols:
            if mol is None:
                continue
                
            # Skip small molecules likely to be catalysts or bases
            num_atoms = mol.GetNumAtoms()
            num_heavy_atoms = mol.GetNumHeavyAtoms()
            
            # Consider fragments with >8 heavy atoms as significant
            # This excludes typical bases, catalysts, solvents
            if num_heavy_atoms > 8:
                significant_count += 1
                
        return significant_count
