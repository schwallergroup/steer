"""Generated evaluation code for: Late stage nucleophilic aromatic substitution coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNucleophilicAromaticSubstitution(BaseScoring):
    """
    Evaluates late-stage nucleophilic aromatic substitution reactions.
    Detects SNAr reactions involving electron-withdrawing groups and nucleophiles,
    with preference for reactions occurring later in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)  # Prefer late stage
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # SNAr reaction doesn't occur
        else:
            # Late-stage reactions are preferred (lower depth fractions are better)
            # Score increases as depth approaches target_depth (0.2 = late stage)
            if self.condition_type == "continuous":
                if x <= self.target_depth:
                    return 1.0  # Perfect score for very late stage
                else:
                    # Penalize reactions that occur too early
                    return max(0, 1.0 - 2 * (x - self.target_depth))
            else:  # bool type
                return 1.0 if x <= self.target_depth else 0.0
    
    def hit_condition(self, d) -> bool:
        """
        Detects nucleophilic aromatic substitution by checking for:
        1. Aromatic system with electron-withdrawing groups
        2. Displacement of leaving groups (halogens, sulfones, nitro)
        3. Formation of C-N, C-O, or C-S bonds to aromatic carbon
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            products = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not products or not all(reactants):
                return False
                
            # Check for aromatic systems with electron-withdrawing groups
            ewg_patterns = [
                "[cH0:1][N+](=O)[O-]",  # nitro on aromatic
                "[cH0:1]C(=O)",         # carbonyl on aromatic  
                "[cH0:1]S(=O)(=O)",     # sulfone on aromatic
                "[cH0:1][N+]",          # quaternary N on aromatic
                "c1[cH0:2]nc[cH0:1]n1", # pyrimidine core
                "c1[cH0:2]nnc[cH0:1]1", # pyrazole core
            ]
            
            # Check for leaving groups being displaced
            leaving_group_patterns = [
                "[cH0:1][Cl,Br,I,F]",        # halogens
                "[cH0:1]S(=O)(=O)[CH3]",     # methyl sulfone
                "[cH0:1][N+](=O)[O-]",       # nitro as leaving group
                "[cH0:1]OS(=O)(=O)",         # sulfonate ester
            ]
            
            # Check for nucleophile attachment (C-N, C-O, C-S formation)
            nucleophile_patterns = [
                "[cH0:1][NH,NH2]",           # amine attachment
                "[cH0:1][OH]",               # hydroxyl attachment  
                "[cH0:1][SH,S]",             # thiol/sulfide attachment
                "[cH0:1]N([CH2,CH3])",       # alkyl amine attachment
            ]
            
            # Check if reactants contain aromatic system with EWG and leaving group
            has_electrophile = False
            for reactant in reactants:
                for ewg_pattern in ewg_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(ewg_pattern)):
                        for lg_pattern in leaving_group_patterns:
                            if reactant.HasSubstructMatch(Chem.MolFromSmarts(lg_pattern)):
                                has_electrophile = True
                                break
                    if has_electrophile:
                        break
                        
            # Check if products show nucleophile attachment to aromatic system
            has_nucleophile_product = False
            for ewg_pattern in ewg_patterns:
                if products.HasSubstructMatch(Chem.MolFromSmarts(ewg_pattern)):
                    for nuc_pattern in nucleophile_patterns:
                        if products.HasSubstructMatch(Chem.MolFromSmarts(nuc_pattern)):
                            has_nucleophile_product = True
                            break
                if has_nucleophile_product:
                    break
                    
            return has_electrophile and has_nucleophile_product
            
        except Exception:
            return False
