"""Generated evaluation code for: Early lactam reduction timing"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyLactamReduction(BaseScoring):
    """
    Evaluates whether lactam reduction occurs early in the synthesis route.
    Detects when a lactam ring (amide in a ring) is reduced to break the C=O bond,
    converting it to an amine. Earlier reduction is scored more favorably.
    """
    
    def __init__(self, config: Dict):
        self.lactam_smarts = config.get("ring_smarts", "[NX3][CX3](=O)")
        self.timing = config.get("timing", "early")
        self.direction = config.get("direction", "break")
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        For early timing: lower depth fractions (earlier) get higher scores.
        """
        if x < 0:
            return 0  # Lactam reduction doesn't occur
        
        if self.timing == "early":
            # Early reduction preferred: score decreases with depth
            return max(0, 10 * (1 - x))
        else:
            # Late reduction preferred: score increases with depth
            return min(10, 10 * x)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves lactam reduction.
        Looks for lactam pattern in product that gets reduced in reactants.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
            
            # Check if product contains lactam pattern
            lactam_pattern = Chem.MolFromSmarts(self.lactam_smarts)
            if not lactam_pattern:
                return False
            
            if not product_mol.HasSubstructMatch(lactam_pattern):
                return False
            
            # Parse reactants
            reactant_mols = []
            for r_smiles in reactants_smiles.split("."):
                r_mol = Chem.MolFromSmiles(r_smiles)
                if r_mol:
                    reactant_mols.append(r_mol)
            
            if not reactant_mols:
                return False
            
            # Check if any reactant has the lactam reduced (no longer matches pattern)
            # This indicates the lactam C=O was reduced
            for reactant in reactant_mols:
                # Check if this reactant corresponds to the lactam-containing region
                # by looking for reduced form (amine instead of amide)
                reduced_lactam_pattern = Chem.MolFromSmarts("[NX3][CX4]")  # N-C single bond instead of N-C=O
                
                if (reactant.HasSubstructMatch(reduced_lactam_pattern) and 
                    not reactant.HasSubstructMatch(lactam_pattern)):
                    # Additional check: ensure we're seeing a ring opening/reduction
                    # by comparing ring counts or looking for typical reduction products
                    if self._is_lactam_reduction(product_mol, reactant):
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _is_lactam_reduction(self, product_mol, reactant_mol) -> bool:
        """
        Helper method to confirm this is actually a lactam reduction reaction.
        Checks for typical signs of amide reduction.
        """
        try:
            # Simple heuristic: reactant should have more hydrogens (from reduction)
            product_h_count = sum(1 for atom in product_mol.GetAtoms() 
                                if atom.GetAtomicNum() == 1)
            reactant_h_count = sum(1 for atom in reactant_mol.GetAtoms() 
                                 if atom.GetAtomicNum() == 1)
            
            # Add implicit hydrogens
            product_h_count += sum(atom.GetTotalNumHs() for atom in product_mol.GetAtoms())
            reactant_h_count += sum(atom.GetTotalNumHs() for atom in reactant_mol.GetAtoms())
            
            # Reduction should add hydrogens
            return reactant_h_count > product_h_count
            
        except Exception:
            return True  # Default to true if we can't determine
