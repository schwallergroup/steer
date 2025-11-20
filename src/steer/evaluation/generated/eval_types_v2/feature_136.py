"""Generated evaluation code for: Late quinoline ring formation via Friedländer annulation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class QuinolineFriedlanderFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage quinoline ring formation via Friedländer annulation.
    Checks for the formation of quinoline rings through condensation of aniline derivatives 
    with enone precursors.
    """
    
    def __init__(self, config):
        self.quinoline_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.method = config["parameters"]["method"]
        
    def route_scoring(self, x):
        """
        Score based on timing of quinoline formation.
        Late-stage formation (higher depth fraction) gets better score.
        """
        if x < 0:
            return 0  # No quinoline formation detected
        
        if self.timing == "late":
            return x * 10  # Reward later formation (depth fraction 0-1 -> score 0-10)
        else:
            return (1 - x) * 10  # Reward earlier formation
    
    def hit_condition(self, d):
        """
        Check if this reaction forms a quinoline ring via Friedländer-type annulation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            # Check if quinoline is formed (present in product but not all reactants)
            if not self._has_quinoline_substructure(products):
                return False
                
            # Check if this is genuinely forming the ring (not just present in starting materials)
            reactant_mols = []
            for r_smi in reactants.split("."):
                r_mol = Chem.MolFromSmiles(r_smi)
                if r_mol:
                    reactant_mols.append(r_mol)
            
            # Quinoline formation: should not be present in individual reactants
            quinoline_in_reactants = any(self._has_quinoline_substructure(Chem.MolToSmiles(mol)) 
                                       for mol in reactant_mols)
            
            if quinoline_in_reactants:
                return False
                
            # Check for Friedländer annulation pattern
            return self._detect_friedlander_pattern(reactant_mols)
            
        except Exception:
            return False
    
    def _has_quinoline_substructure(self, smiles):
        """Check if molecule contains quinoline substructure."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return False
            quinoline_pattern = Chem.MolFromSmarts(self.quinoline_smarts)
            if not quinoline_pattern:
                return False
            return mol.HasSubstructMatch(quinoline_pattern)
        except Exception:
            return False
    
    def _detect_friedlander_pattern(self, reactant_mols):
        """
        Detect Friedländer annulation pattern:
        - One reactant should contain aniline-like structure (aromatic amine)
        - Another should contain enone or carbonyl that can form the pyridine ring
        """
        if len(reactant_mols) < 2:
            return False
            
        # Aniline-like pattern: aromatic ring with amino group
        aniline_pattern = Chem.MolFromSmarts("c1ccccc1N")
        # Extended aniline patterns
        aniline_patterns = [
            Chem.MolFromSmarts("c1ccccc1N"),  # basic aniline
            Chem.MolFromSmarts("c1ccc(N)cc1"),  # para-substituted aniline
            Chem.MolFromSmarts("c1cc(N)ccc1"),  # meta-substituted aniline
            Chem.MolFromSmarts("c1c(N)cccc1"),  # ortho-substituted aniline
        ]
        
        # Enone/carbonyl patterns that could form pyridine ring
        carbonyl_patterns = [
            Chem.MolFromSmarts("C=O"),  # general carbonyl
            Chem.MolFromSmarts("CC=CC=O"),  # enone
            Chem.MolFromSmarts("C=CC(=O)C"),  # enone variant
            Chem.MolFromSmarts("C(=O)CC=O"),  # dicarbonyl
        ]
        
        has_aniline = False
        has_carbonyl = False
        
        for mol in reactant_mols:
            # Check for aniline-like structure
            if any(mol.HasSubstructMatch(pattern) for pattern in aniline_patterns if pattern):
                has_aniline = True
                
            # Check for carbonyl/enone structure
            if any(mol.HasSubstructMatch(pattern) for pattern in carbonyl_patterns if pattern):
                has_carbonyl = True
        
        return has_aniline and has_carbonyl
