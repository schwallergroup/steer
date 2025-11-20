"""Generated evaluation code for: Late stage SNAr coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAromaticSubstitution(BaseScoring):
    """
    Evaluates whether a nucleophilic aromatic substitution (SNAr) reaction 
    occurs in the final step of the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.target_stage = config.get("stage", "final_step")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # SNAr doesn't happen
        elif x == 0:
            return 10  # Perfect - happens in final step
        else:
            return max(0, 10 - x * 5)  # Penalty for earlier stages
    
    def hit_condition(self, d):
        """Check if this reaction is a nucleophilic aromatic substitution"""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            product = Chem.MolFromSmiles(rxn[0])
            
            # Check for SNAr pattern: aromatic carbon with leaving group -> aromatic C-N bond
            return self._is_snar_reaction(reactants, product)
            
        except:
            return False
    
    def _is_snar_reaction(self, reactants, product):
        """Detect SNAr by looking for aromatic C-N bond formation with leaving groups"""
        # Pattern for electron-withdrawing aromatic systems (common SNAr substrates)
        ew_aromatic_patterns = [
            "[cH0:1]1[c]([N+](=O)[O-])[c][c][c][c]1",  # nitrobenzene derivatives
            "[cH0:1]1[c]([C](=O))[c][c][c][c]1",        # benzoyl derivatives
            "[cH0:1]1[c]([C]#N)[c][c][c][c]1",          # benzonitrile derivatives
            "[cH0:1]1[c]([S](=O)(=O))[c][c][c][c]1",    # sulfonyl derivatives
        ]
        
        # Common leaving groups in SNAr
        leaving_groups = [
            "[S](=O)[CH3]",  # methylsulfinyl (as mentioned in rationale)
            "[Cl]",          # chloride
            "[F]",           # fluoride
            "[N+]",          # nitro group as leaving group
            "[S]",           # sulfur-based leaving groups
        ]
        
        # Look for nucleophile (amine) in reactants
        nucleophile_patterns = [
            "[NH2]",         # primary amine
            "[NH1]",         # secondary amine
            "[nH]",          # aromatic amine
        ]
        
        # Check if reactants contain aromatic substrate with leaving group
        has_aromatic_substrate = False
        has_nucleophile = False
        has_leaving_group = False
        
        for reactant in reactants:
            # Check for electron-withdrawing aromatic substrate
            for pattern in ew_aromatic_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_aromatic_substrate = True
                    break
            
            # Check for leaving groups
            for lg_pattern in leaving_groups:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(lg_pattern)):
                    has_leaving_group = True
                    break
                    
            # Check for nucleophile
            for nuc_pattern in nucleophile_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(nuc_pattern)):
                    has_nucleophile = True
                    break
        
        # Check if product has aromatic C-N bond
        aromatic_cn_patterns = [
            "[c][NH2]",      # aromatic primary amine
            "[c][NH1]",      # aromatic secondary amine  
            "[c][N]([C])",   # aromatic tertiary amine
            "[c][nH]",       # aromatic nitrogen in ring
        ]
        
        has_aromatic_cn = False
        for pattern in aromatic_cn_patterns:
            if product.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                has_aromatic_cn = True
                break
        
        return has_aromatic_substrate and has_nucleophile and has_aromatic_cn
