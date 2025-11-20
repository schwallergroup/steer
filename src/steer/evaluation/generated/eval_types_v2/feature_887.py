"""Generated evaluation code for: Late piperazine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePiperazineFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage piperazine ring formation.
    Detects when a piperazine ring (C1CNCCN1) is formed late in the synthesis,
    typically via double N-alkylation cyclization reactions.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config.get("ring_smarts", "C1CNCCN1")
        self.timing = config.get("timing", "late")
        self.direction = config.get("direction", "formation")
        
    def route_scoring(self, x) -> float:
        """
        Score based on timing of piperazine ring formation.
        Late formation (higher depth fraction) gets better score.
        """
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # Late-stage formation is rewarded (closer to 1.0 depth fraction)
            return x * 10  # Convert to 0-10 scale, higher is better for late timing
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves piperazine ring formation.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        # Parse reactants and product
        reactants_smiles = rxn_parts[0]
        product_smiles = rxn_parts[1]
        
        try:
            # Check product has piperazine ring
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
                
            piperazine_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if not product_mol.HasSubstructMatch(piperazine_pattern):
                return False
            
            # Check that reactants don't have the complete piperazine ring
            reactant_mols = []
            for reactant_smiles in reactants_smiles.split("."):
                reactant_mol = Chem.MolFromSmiles(reactant_smiles.strip())
                if reactant_mol:
                    reactant_mols.append(reactant_mol)
            
            # If any reactant already has the piperazine ring, this isn't formation
            for reactant_mol in reactant_mols:
                if reactant_mol.HasSubstructMatch(piperazine_pattern):
                    return False
            
            # Additional check: look for N-alkylation pattern typical of piperazine formation
            # This helps distinguish true ring formation from other transformations
            return self._detect_cyclization_pattern(reactant_mols, product_mol)
            
        except Exception:
            return False
    
    def _detect_cyclization_pattern(self, reactants, product):
        """
        Helper method to detect if this looks like a cyclization reaction
        that would form a piperazine ring (e.g., intramolecular double N-alkylation).
        """
        # Look for precursor patterns that could cyclize to form piperazine
        # Common patterns: diamine with dihalide, or linear diamine with leaving groups
        linear_diamine_pattern = Chem.MolFromSmarts("NCCNCC")  # Extended chain
        dihalide_pattern = Chem.MolFromSmarts("[Cl,Br,I]CC[Cl,Br,I]")  # Dihalide
        
        has_diamine = False
        has_electrophile = False
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(linear_diamine_pattern):
                has_diamine = True
            if reactant.HasSubstructMatch(dihalide_pattern):
                has_electrophile = True
        
        # If we have both components, this looks like piperazine formation
        return has_diamine or has_electrophile
