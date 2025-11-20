"""Generated evaluation code for: Late stage pyridine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates synthesis routes based on late-stage formation of specific ring systems.
    Checks for pyridine ring formation via Skraup-type annulation reactions.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.formation_method = config["parameters"]["formation_method"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
        # SMARTS patterns for Skraup-type reactions
        # Aniline derivative + carbonyl compound forming pyridine
        self.skraup_patterns = [
            "[#7;H2,H1:1]-[c:2]1[c:3][c:4][c:5][c:6][c:7]1>>[n:1]1[c:2][c:3][c:4][c:5][c:6]1",
            "[#7:1]-[c:2]1[c:3][c:4][c:5][c:6][c:7]1.[C:8]=[O:9]>>[n:1]1[c:2][c:3][c:4][c:5][c:8]1",
            "[c:1]1[c:2][c:3]([NH2:4])[c:5][c:6][c:7]1.[C:8]=[C:9]>>[c:1]1[c:2][c:3]2[n:4][c:8][c:9][c:5][c:6]2[c:7]1"
        ]

    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        For late-stage formation, higher depth (later) is better.
        """
        if x < 0:
            return 0  # Ring formation doesn't occur
        
        if self.timing == "late":
            # Late-stage preferred: higher depth gets higher score
            return min(10, x * 10)
        elif self.timing == "early":
            # Early-stage preferred: lower depth gets higher score  
            return max(0, 10 - x * 10)
        else:
            # Any timing acceptable
            return 5 if x >= 0 else 0

    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents pyridine ring formation via Skraup-type reaction.
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
            product_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
                
            # Check if product contains the target ring
            if not product_mol.HasSubstructMatch(self.ring_pattern):
                return False
                
            # Check if this is ring formation (not just present in starting material)
            reactant_smiles_list = reactants_smiles.split(".")
            reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactant_smiles_list if smi]
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            
            # If any reactant already has the ring, this is not ring formation
            for reactant in reactant_mols:
                if reactant.HasSubstructMatch(self.ring_pattern):
                    return False
            
            # Check for Skraup-type reaction pattern if specified
            if self.formation_method == "Skraup_type":
                return self._is_skraup_type_reaction(reactant_mols, product_mol)
            
            # If no specific method required, any ring formation counts
            return True
            
        except Exception:
            return False

    def _is_skraup_type_reaction(self, reactants, product) -> bool:
        """
        Check if the reaction follows Skraup-type annulation patterns.
        """
        # Look for aniline derivative in reactants
        aniline_pattern = Chem.MolFromSmarts("c1ccc(N)cc1")  # Basic aniline
        aniline_subst_pattern = Chem.MolFromSmarts("c1ccc([NH2,NH1])cc1")  # Substituted anilines
        
        has_aniline = any(
            reactant.HasSubstructMatch(aniline_pattern) or 
            reactant.HasSubstructMatch(aniline_subst_pattern)
            for reactant in reactants
        )
        
        if not has_aniline:
            return False
            
        # Look for carbonyl or alkene components that could cyclize
        carbonyl_pattern = Chem.MolFromSmarts("[C]=[O]")
        alkene_pattern = Chem.MolFromSmarts("[C]=[C]")
        
        has_electrophile = any(
            reactant.HasSubstructMatch(carbonyl_pattern) or
            reactant.HasSubstructMatch(alkene_pattern)
            for reactant in reactants
        )
        
        return has_electrophile
