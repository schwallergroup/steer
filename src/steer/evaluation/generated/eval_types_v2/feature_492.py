"""Generated evaluation code for: Late stage nucleophilic aromatic substitution"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNucleophilicAromaticSubstitution(BaseScoring):
    """
    Evaluates whether nucleophilic aromatic substitution (SNAr) occurs as a late-stage reaction.
    Returns higher scores when SNAr happens closer to the final step.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "late")
        self.step_position = config.get("step_position", "final")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # SNAr reaction doesn't occur
        else:
            # Higher score for later occurrence (lower depth fraction)
            # Final step (x=0) gets score of 1, earlier steps get lower scores
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Detects nucleophilic aromatic substitution reactions by checking for:
        1. Formation of C-N bond to aromatic carbon
        2. Presence of electron-withdrawing groups on aromatic ring
        3. Loss of leaving group from aromatic carbon
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            return self._detect_snar_pattern(product, reactants)
            
        except Exception:
            return False
    
    def _detect_snar_pattern(self, product, reactants) -> bool:
        """
        Detects SNAr by looking for:
        - Aromatic C-N bond formation
        - Electron-withdrawing groups on aromatic ring
        - Nucleophile in reactants
        """
        # Pattern for aromatic C-N bond (nitrogen attached to aromatic carbon)
        aromatic_cn_pattern = Chem.MolFromSmarts("[c][N]")
        
        # Check if product has aromatic C-N bond
        if not product.HasSubstructMatch(aromatic_cn_pattern):
            return False
        
        # Look for nucleophilic nitrogen in reactants
        nucleophile_patterns = [
            Chem.MolFromSmarts("[N;H2]"),  # Primary amine
            Chem.MolFromSmarts("[N;H1]"),  # Secondary amine
            Chem.MolFromSmarts("[N;H0;!$(N=*);!$(N#*)]"),  # Tertiary amine
            Chem.MolFromSmarts("[NH-]"),   # Amide anion
        ]
        
        has_nucleophile = any(
            any(reactant.HasSubstructMatch(pattern) for pattern in nucleophile_patterns)
            for reactant in reactants
        )
        
        if not has_nucleophile:
            return False
        
        # Check for electron-withdrawing groups on aromatic ring
        ewg_patterns = [
            Chem.MolFromSmarts("[c][N+](=O)[O-]"),  # Nitro group
            Chem.MolFromSmarts("[c][C](=O)"),       # Carbonyl
            Chem.MolFromSmarts("[c][C#N]"),         # Cyano group
            Chem.MolFromSmarts("[c][S](=O)(=O)"),   # Sulfonyl
            Chem.MolFromSmarts("[c][C](F)(F)F"),    # Trifluoromethyl
        ]
        
        has_ewg = any(product.HasSubstructMatch(pattern) for pattern in ewg_patterns)
        
        # Also check for common leaving groups in reactants (halides)
        leaving_group_patterns = [
            Chem.MolFromSmarts("[c][Cl]"),  # Aryl chloride
            Chem.MolFromSmarts("[c][Br]"),  # Aryl bromide  
            Chem.MolFromSmarts("[c][I]"),   # Aryl iodide
            Chem.MolFromSmarts("[c][F]"),   # Aryl fluoride
        ]
        
        has_leaving_group = any(
            any(reactant.HasSubstructMatch(pattern) for pattern in leaving_group_patterns)
            for reactant in reactants
        )
        
        # SNAr is likely if we have EWG or obvious leaving group
        return has_ewg or has_leaving_group
