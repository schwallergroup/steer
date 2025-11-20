"""Generated evaluation code for: Early pyrimidine ring formation via cyclocondensation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyPyrimidineFormation(BaseScoring):
    """
    Evaluates whether pyrimidine ring formation via cyclocondensation occurs early in the synthesis route.
    Detects formation of pyrimidine rings (c1ncncn1) through cyclocondensation reactions and scores
    based on how early this occurs in the route.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1ncncn1"
        self.timing = config["parameters"]["timing"]  # "early"
        self.formation_method = config["parameters"]["formation_method"]  # "cyclocondensation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Pyrimidine formation doesn't happen
        else:
            if self.timing == "early":
                return 1 - x  # Earlier formation gets higher score
            else:
                return x  # Later formation gets higher score
    
    def hit_condition(self, d):
        """Check if this reaction forms a pyrimidine ring via cyclocondensation"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if product contains pyrimidine ring
            if not product_mol.HasSubstructMatch(self.ring_pattern):
                return False
            
            # Check if any reactant already contains pyrimidine ring
            for reactant in reactant_mols:
                if reactant.HasSubstructMatch(self.ring_pattern):
                    return False  # Ring already exists, not a formation reaction
            
            # Check for cyclocondensation pattern
            # Look for nitrogen-containing reactants that could cyclize
            return self._is_cyclocondensation(reactant_mols, product_mol)
            
        except Exception:
            return False
    
    def _is_cyclocondensation(self, reactant_mols, product_mol):
        """Check if the reaction pattern matches cyclocondensation"""
        # Count nitrogen atoms in reactants vs product
        reactant_n_count = sum(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7) 
                              for mol in reactant_mols)
        product_n_count = sum(1 for atom in product_mol.GetAtoms() if atom.GetAtomicNum() == 7)
        
        # For pyrimidine formation, we expect nitrogen-containing precursors
        if reactant_n_count < 2:
            return False
        
        # Check for typical cyclocondensation patterns
        # Look for imine (C=N) and amide/thioamide patterns in reactants
        imine_pattern = Chem.MolFromSmarts("C=N")
        amide_pattern = Chem.MolFromSmarts("C(=O)N")
        thioamide_pattern = Chem.MolFromSmarts("C(=S)N")
        
        has_imine = any(mol.HasSubstructMatch(imine_pattern) for mol in reactant_mols)
        has_amide_like = any(mol.HasSubstructMatch(amide_pattern) or 
                           mol.HasSubstructMatch(thioamide_pattern) 
                           for mol in reactant_mols)
        
        return has_imine and has_amide_like
