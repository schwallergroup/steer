"""Generated evaluation code for: Cbz protecting group strategy for piperidine amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CbzPiperidineProtection(BaseScoring):
    """
    Evaluates synthesis routes for Cbz (carboxybenzyl) protecting group strategy 
    on piperidine secondary amines. Checks for presence of Cbz-protected piperidine
    intermediates and their subsequent deprotection via hydrogenolysis.
    
    Returns higher scores when Cbz protection occurs earlier in the synthesis,
    allowing for selective manipulation of other functional groups.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.3)
        
        # SMARTS patterns for detection
        self.cbz_piperidine_pattern = "[CH2]([c]1[cH][cH][cH][cH][cH]1)[O][C](=[O])[N]1[CH2][CH2][CH2][CH2][CH2]1"
        self.free_piperidine_pattern = "[NH]1[CH2][CH2][CH2][CH2][CH2]1"
        self.cbz_pattern = "[CH2]([c]1[cH][cH][cH][cH][cH]1)[O][C](=[O])[NH]"
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to 0-10 score, favoring early protection"""
        if x < 0:
            return 0  # Protection strategy not found
        
        if self.condition_type == "bool":
            return 10 if x >= 0 else 0
        else:
            # Early protection (low depth) gets higher score
            return max(0, 10 - (x * 10))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves Cbz protection of a piperidine amine
        or formation of Cbz-protected piperidine intermediate
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse molecules
            reactants = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi.strip())
                if mol:
                    reactants.append(mol)
            
            products = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi.strip())
                if mol:
                    products.append(mol)
            
            # Check for Cbz protection: free piperidine + Cbz reagent -> Cbz-protected piperidine
            has_free_piperidine_reactant = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.free_piperidine_pattern))
                for mol in reactants
            )
            
            has_cbz_reagent = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts("[CH2]([c]1[cH][cH][cH][cH][cH]1)[O][C](=[O])[Cl]")) or
                mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)([O][CH2][c]1[cH][cH][cH][cH][cH]1)[O][C](=O)"))
                for mol in reactants
            )
            
            has_cbz_piperidine_product = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.cbz_piperidine_pattern))
                for mol in products
            )
            
            # Check for formation of Cbz-protected piperidine intermediate
            cbz_piperidine_formed = (has_free_piperidine_reactant and has_cbz_reagent and has_cbz_piperidine_product)
            
            # Also check for reactions that form Cbz-protected products (even without explicit reagent)
            cbz_protection_reaction = (
                not any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.cbz_piperidine_pattern)) for mol in reactants) and
                has_cbz_piperidine_product
            )
            
            return cbz_piperidine_formed or cbz_protection_reaction
            
        except Exception:
            return False
