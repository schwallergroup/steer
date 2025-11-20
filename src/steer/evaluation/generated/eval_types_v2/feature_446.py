"""Generated evaluation code for: Late pyrimidine ring formation via Traube cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePyrimidineTraube(BaseScoring):
    """
    Evaluates synthesis routes for late-stage pyrimidine ring formation via Traube cyclization.
    Checks if a pyrimidine ring (c1ncncc1) is formed late in the synthesis using formamide 
    cyclization with ortho-amino nitrile precursors.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.method = config["parameters"]["method"]
        self.pyrimidine_pattern = Chem.MolFromSmarts(self.ring_smarts)
        # Pattern for ortho-amino nitrile (precursor for Traube cyclization)
        self.amino_nitrile_pattern = Chem.MolFromSmarts("[NH2]c1ccccc1C#N")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Pyrimidine ring formation not found
        
        if self.timing == "late":
            # Late formation preferred - higher score for higher depth fraction
            return 10 * x
        else:
            # Early formation preferred - higher score for lower depth fraction  
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents pyrimidine ring formation via Traube cyclization.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            product_smiles, reactant_smiles = mapped_rxn.split(">>")
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if pyrimidine ring is formed (present in product but not in reactants)
            product_has_pyrimidine = product_mol.HasSubstructMatch(self.pyrimidine_pattern)
            reactants_have_pyrimidine = any(mol.HasSubstructMatch(self.pyrimidine_pattern) for mol in reactant_mols)
            
            if not (product_has_pyrimidine and not reactants_have_pyrimidine):
                return False
            
            # Check for Traube cyclization signature: ortho-amino nitrile precursor
            has_amino_nitrile_precursor = any(mol.HasSubstructMatch(self.amino_nitrile_pattern) for mol in reactant_mols)
            
            # Check for formamide or formic acid derivatives (common cyclization agents)
            formamide_pattern = Chem.MolFromSmarts("NC=O")
            formic_pattern = Chem.MolFromSmarts("C(=O)O")
            has_formamide = any(mol.HasSubstructMatch(formamide_pattern) for mol in reactant_mols)
            has_formic = any(mol.HasSubstructMatch(formic_pattern) for mol in reactant_mols)
            
            return has_amino_nitrile_precursor and (has_formamide or has_formic)
            
        except Exception:
            return False
