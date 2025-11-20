"""Generated evaluation code for: Late triazole ring formation via cycloaddition"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TriazoleRingFormation(BaseScoring):
    """
    Evaluates late-stage triazole ring formation via cycloaddition reactions.
    Detects formation of triazole rings (n1nncn1) through cycloaddition and 
    scores based on how late in the synthesis this occurs.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.formation_method = config["parameters"]["formation_method"]
        self.timing = config["parameters"]["timing"]
        self.triazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Triazole formation doesn't happen
        else:
            # Late-stage formation is better for "late" timing
            if self.timing == "late":
                return 1 - x  # Higher score for later formation
            else:
                return x  # Higher score for earlier formation
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents triazole ring formation via cycloaddition.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if product contains triazole ring
            if not product_mol.HasSubstructMatch(self.triazole_pattern):
                return False
            
            # Check if any reactant already contains triazole ring
            for reactant in reactant_mols:
                if reactant.HasSubstructMatch(self.triazole_pattern):
                    return False  # Triazole already present, not formation
            
            # Check for cycloaddition characteristics
            if self.formation_method == "cycloaddition":
                return self._is_cycloaddition_reaction(product_mol, reactant_mols)
            
            return True
            
        except Exception:
            return False
    
    def _is_cycloaddition_reaction(self, product_mol, reactant_mols) -> bool:
        """
        Check if reaction characteristics match cycloaddition (e.g., [3+2] for triazole).
        Look for typical cycloaddition patterns: alkyne + azide/azomethine imine.
        """
        # Check for alkyne in reactants
        alkyne_pattern = Chem.MolFromSmarts("C#C")
        has_alkyne = any(r.HasSubstructMatch(alkyne_pattern) for r in reactant_mols)
        
        # Check for azide or azomethine imine patterns
        azide_pattern = Chem.MolFromSmarts("N=[N+]=[N-]")  # Azide
        azomethine_pattern = Chem.MolFromSmarts("C=N-N")    # Azomethine imine
        
        has_dipole = any(r.HasSubstructMatch(azide_pattern) or 
                        r.HasSubstructMatch(azomethine_pattern) for r in reactant_mols)
        
        # Typical triazole cycloaddition involves alkyne + nitrogen dipole
        return has_alkyne and has_dipole
