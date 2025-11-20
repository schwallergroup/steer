"""Generated evaluation code for: Late stage nucleophilic aromatic substitution"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNucleophilicAromaticSubstitution(BaseScoring):
    """
    Evaluates if nucleophilic aromatic substitution (SNAr) occurs at late stage.
    
    SNAr reactions involve nucleophilic attack on electron-deficient aromatic rings,
    typically containing electron-withdrawing groups like NO2, CN, or halogens in
    ortho/para positions. Late-stage SNAr is valued for strategic fragment coupling.
    """
    
    def __init__(self, config: Dict):
        self.stage = config.get("stage", "final")  # "final" or "late"
        self.coupling_step = config.get("coupling_step", True)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # SNAr doesn't happen
        else:
            if self.stage == "final":
                # Reward very late stage (closer to 1.0 is better)
                return 10 * (1 - x) if x > 0.7 else 0
            else:  # late stage
                return 10 * (1 - x) if x > 0.5 else 0
                
    def hit_condition(self, d) -> bool:
        """Check if reaction is nucleophilic aromatic substitution"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".") if r]
            
            if not prod_mol or not react_mols:
                return False
                
            # Check for SNAr pattern: aromatic substitution with nucleophile
            return self._is_snar_reaction(prod_mol, react_mols)
            
        except:
            return False
            
    def _is_snar_reaction(self, product, reactants) -> bool:
        """Detect SNAr by checking for aromatic substitution patterns"""
        
        # Look for electron-deficient aromatic rings in product
        ew_aromatic_patterns = [
            "[cH0:1][c:2][N+](=O)[O-]",  # nitro-substituted aromatic
            "[cH0:1][c:2][C]#[N]",       # cyano-substituted aromatic  
            "[cH0:1][c:2][C](=O)",       # carbonyl-substituted aromatic
            "[cH0:1][c:2][S](=O)(=O)",   # sulfonyl-substituted aromatic
            "[cH0:1][c:2][F,Cl,Br,I]",   # halogen-substituted aromatic
        ]
        
        # Check if product contains electron-withdrawing aromatic system
        has_ew_aromatic = False
        for pattern in ew_aromatic_patterns:
            if product.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                has_ew_aromatic = True
                break
                
        if not has_ew_aromatic:
            return False
            
        # Look for nucleophile in reactants
        nucleophile_patterns = [
            "[NH2,NH,N]",      # amines
            "[OH]",            # alcohols/phenols
            "[SH,S-]",         # thiols/sulfides
            "[O-]",            # alkoxides
            "[N-]",            # amides
            "[C-]",            # carbanions
        ]
        
        has_nucleophile = False
        for reactant in reactants:
            for nuc_pattern in nucleophile_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(nuc_pattern)):
                    has_nucleophile = True
                    break
            if has_nucleophile:
                break
                
        # Check for leaving group in reactants (halogen typically)
        leaving_group_pattern = "[F,Cl,Br,I,N+]"
        has_leaving_group = any(
            reactant.HasSubstructMatch(Chem.MolFromSmarts(leaving_group_pattern)) 
            for reactant in reactants
        )
        
        # SNAr typically involves nucleophile + electrophilic aromatic with leaving group
        if self.coupling_step:
            # For coupling, expect 2+ reactants coming together
            return (len(reactants) >= 2 and has_nucleophile and 
                    has_ew_aromatic and has_leaving_group)
        else:
            return has_nucleophile and has_ew_aromatic
