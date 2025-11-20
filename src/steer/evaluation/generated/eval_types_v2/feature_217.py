"""Generated evaluation code for: Late stage ether formation via Williamson synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageWilliamsonEther(BaseScoring):
    """
    Evaluates routes for late-stage ether formation via Williamson synthesis.
    Detects C-O ether bond formation through nucleophilic substitution between 
    phenol and alkyl halide, rewarding later occurrence in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.bond_type = config.get("bond_type", "C-O")
        self.reaction_type = config.get("reaction_type", "Williamson_ether_synthesis")
        self.timing = config.get("timing", "late")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Williamson ether synthesis doesn't occur
        else:
            # Late-stage ether formation is better, so invert depth fraction
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Detect Williamson ether synthesis by checking for:
        1. Formation of C-O ether bond
        2. Presence of phenol nucleophile
        3. Presence of alkyl halide electrophile
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            product_smiles, reactant_smiles = rxn_smiles.split(">>")
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for ether formation (C-O-C pattern)
            ether_pattern = Chem.MolFromSmarts("[C]-[O]-[C]")
            if not product.HasSubstructMatch(ether_pattern):
                return False
            
            # Check for phenol reactant (aromatic OH)
            phenol_pattern = Chem.MolFromSmarts("[OH]-[c]")
            has_phenol = any(r.HasSubstructMatch(phenol_pattern) for r in reactants)
            
            # Check for alkyl halide reactant (C-X where X is halogen)
            alkyl_halide_pattern = Chem.MolFromSmarts("[C]-[F,Cl,Br,I]")
            has_alkyl_halide = any(r.HasSubstructMatch(alkyl_halide_pattern) for r in reactants)
            
            # Verify C-O bond formation by checking atom mapping
            if has_phenol and has_alkyl_halide:
                return self._verify_co_bond_formation(product, reactants)
            
            return False
            
        except Exception:
            return False
    
    def _verify_co_bond_formation(self, product, reactants) -> bool:
        """
        Verify that a new C-O bond is formed between carbon from alkyl halide
        and oxygen from phenol using atom mapping.
        """
        try:
            # Get atom map numbers for ether oxygens in product
            ether_pattern = Chem.MolFromSmarts("[C]-[O]-[C]")
            matches = product.GetSubstructMatches(ether_pattern)
            
            for match in matches:
                c1_idx, o_idx, c2_idx = match
                o_mapnum = product.GetAtomWithIdx(o_idx).GetAtomMapNum()
                c1_mapnum = product.GetAtomWithIdx(c1_idx).GetAtomMapNum()
                c2_mapnum = product.GetAtomWithIdx(c2_idx).GetAtomMapNum()
                
                if o_mapnum == 0 or (c1_mapnum == 0 and c2_mapnum == 0):
                    continue
                
                # Check if oxygen comes from phenol and carbon from alkyl halide
                o_in_phenol = False
                c_in_halide = False
                
                for reactant in reactants:
                    # Check if oxygen is from phenol
                    phenol_pattern = Chem.MolFromSmarts("[OH]-[c]")
                    phenol_matches = reactant.GetSubstructMatches(phenol_pattern)
                    for p_match in phenol_matches:
                        o_atom = reactant.GetAtomWithIdx(p_match[0])
                        if o_atom.GetAtomMapNum() == o_mapnum:
                            o_in_phenol = True
                    
                    # Check if carbon is from alkyl halide
                    halide_pattern = Chem.MolFromSmarts("[C]-[F,Cl,Br,I]")
                    halide_matches = reactant.GetSubstructMatches(halide_pattern)
                    for h_match in halide_matches:
                        c_atom = reactant.GetAtomWithIdx(h_match[0])
                        if c_atom.GetAtomMapNum() in [c1_mapnum, c2_mapnum]:
                            c_in_halide = True
                
                if o_in_phenol and c_in_halide:
                    return True
            
            return False
            
        except Exception:
            return False
