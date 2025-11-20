"""Generated evaluation code for: Late stage ether formation via SNAr"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageEtherSNAr(BaseScoring):
    """
    Evaluates whether late-stage ether formation via nucleophilic aromatic substitution (SNAr) occurs.
    Checks for C-O bond formation through SNAr mechanism at late stages of synthesis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # SNAr ether formation doesn't happen
        else:
            # Late-stage reactions are better - higher x values get higher scores
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves SNAr ether formation.
        Looks for:
        1. Aromatic C-O bond formation
        2. Leaving group displacement pattern
        3. Electron-withdrawing groups on aromatic ring
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Look for nucleophile with OH group (alcohol/phenol)
            nucleophile_pattern = Chem.MolFromSmarts("[OH]")
            has_nucleophile = any(r.HasSubstructMatch(nucleophile_pattern) for r in reactants)
            
            # Look for aromatic halide or other leaving group
            electrophile_patterns = [
                Chem.MolFromSmarts("c[F,Cl,Br,I]"),  # Aromatic halides
                Chem.MolFromSmarts("c[N+](=O)[O-]"),  # Nitro as leaving group
                Chem.MolFromSmarts("cS(=O)(=O)[CH3]")  # Tosylate-like
            ]
            
            has_electrophile = any(
                any(r.HasSubstructMatch(pattern) for r in reactants)
                for pattern in electrophile_patterns
            )
            
            # Check for aromatic ether formation in product
            aromatic_ether_pattern = Chem.MolFromSmarts("cO[!c]")  # Aromatic C-O-aliphatic
            phenyl_ether_pattern = Chem.MolFromSmarts("cOc")      # Aromatic C-O-aromatic
            
            has_aromatic_ether = (product.HasSubstructMatch(aromatic_ether_pattern) or 
                                product.HasSubstructMatch(phenyl_ether_pattern))
            
            # Look for electron-withdrawing groups that activate SNAr
            ewg_patterns = [
                Chem.MolFromSmarts("c[N+](=O)[O-]"),  # Nitro
                Chem.MolFromSmarts("cC(=O)[#6,#1]"),  # Carbonyl
                Chem.MolFromSmarts("cC#N"),           # Cyano
                Chem.MolFromSmarts("cC(F)(F)F"),      # Trifluoromethyl
                Chem.MolFromSmarts("cS(=O)(=O)[#6]")  # Sulfonyl
            ]
            
            has_ewg = any(
                product.HasSubstructMatch(pattern) or 
                any(r.HasSubstructMatch(pattern) for r in reactants)
                for pattern in ewg_patterns
            )
            
            return has_nucleophile and has_electrophile and has_aromatic_ether and has_ewg
            
        except Exception:
            return False
