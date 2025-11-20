"""Generated evaluation code for: Late stage aromatic substitution for functionalization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAromaticSubstitution(BaseScoring):
    """
    Evaluates routes for late-stage nucleophilic aromatic substitution (SNAr) reactions.
    
    Rewards routes where SNAr reactions occur late in the synthesis (high depth fraction),
    typically used for final functionalization steps on pre-formed aromatic cores.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config.get("stage_threshold", 0.8)
    
    def route_scoring(self, x) -> float:
        """
        Score based on how late the SNAr reaction occurs.
        Later reactions (higher depth fraction) get better scores.
        """
        if x < 0:
            return 0  # No SNAr reaction found
        
        # Reward late-stage reactions above threshold
        if x >= self.stage_threshold:
            return 10  # Perfect score for very late stage
        else:
            # Linearly scale score based on depth fraction
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Detect nucleophilic aromatic substitution reactions.
        
        Looks for:
        1. Aromatic ring with electron-withdrawing groups
        2. Nucleophile attacking aromatic carbon
        3. Leaving group displacement pattern
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            return self._detect_snar_pattern(reactants, products)
            
        except Exception:
            return False
    
    def _detect_snar_pattern(self, reactants, products) -> bool:
        """
        Detect SNAr reaction pattern by looking for:
        - Aromatic substrate with electron-withdrawing groups
        - Nucleophile addition to aromatic ring
        - Leaving group departure
        """
        # Common SNAr substrate patterns (electron-deficient aromatics)
        snar_substrate_patterns = [
            # Pyridine derivatives
            "[cH:1]1[c:2][c:3][n:4][c:5][c:6]1",  # pyridine ring
            # Nitroaromatics
            "[c:1]1[c:2][c:3]([N+](=O)[O-])[c:4][c:5][c:6]1",  # nitrobenzene
            # Cyanoaromatics  
            "[c:1]1[c:2][c:3](C#N)[c:4][c:5][c:6]1",  # cyanobenzene
            # Halopyridines (common SNAr substrates)
            "[c:1]1[c:2][c:3]([Cl,F,Br])[n:4][c:5][c:6]1",  # halopyridine
        ]
        
        # Common nucleophiles in SNAr
        nucleophile_patterns = [
            "[NH2:1]",  # primary amine
            "[NH:1]",   # secondary amine  
            "[N:1]",    # tertiary amine
            "[OH:1]",   # hydroxide/alcohol
            "[SH:1]",   # thiol
            "[S:1]",    # thiolate
        ]
        
        # Look for electron-deficient aromatic in reactants
        has_snar_substrate = False
        for reactant in reactants:
            for pattern in snar_substrate_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_snar_substrate = True
                    break
            if has_snar_substrate:
                break
        
        # Look for nucleophile in reactants
        has_nucleophile = False
        for reactant in reactants:
            for pattern in nucleophile_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_nucleophile = True
                    break
            if has_nucleophile:
                break
        
        # Additional check: look for aromatic C-N, C-O, or C-S bond formation
        aromatic_substitution_products = [
            "[c:1][NH:2]",  # aromatic C-N
            "[c:1][OH:2]",  # aromatic C-O  
            "[c:1][SH:2]",  # aromatic C-S
            "[c:1][N:2]",   # aromatic C-N (general)
        ]
        
        has_aromatic_substitution = False
        for product in products:
            for pattern in aromatic_substitution_products:
                if product.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_aromatic_substitution = True
                    break
            if has_aromatic_substitution:
                break
        
        return has_snar_substrate and has_nucleophile and has_aromatic_substitution
