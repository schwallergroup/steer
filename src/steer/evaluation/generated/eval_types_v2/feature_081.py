"""Generated evaluation code for: Late stage nucleophilic aromatic substitution"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNucleophilicAromaticSubstitution(BaseScoring):
    """
    Evaluates routes for late-stage nucleophilic aromatic substitution reactions.
    
    This class checks if a nucleophilic aromatic substitution (SNAr) reaction
    occurs late in the synthesis route, particularly looking for patterns where
    an electron-deficient aromatic system undergoes nucleophilic attack.
    """
    
    def __init__(self, config: Dict):
        self.step_position = config["parameters"].get("step_position", 1)
        
    def route_scoring(self, x) -> float:
        """
        Converts depth fraction to score (0-10).
        Late-stage reactions (high x values) get higher scores.
        """
        if x < 0:
            return 0  # SNAr doesn't occur
        else:
            # Late-stage reactions are preferred, so higher x gives higher score
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Checks if the reaction node represents a nucleophilic aromatic substitution.
        
        Detects SNAr by looking for:
        1. Aromatic carbon with electron-withdrawing groups
        2. Loss of a leaving group (halide, nitro, etc.)
        3. Formation of new C-N, C-O, or C-S bonds to aromatic carbon
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".") if p.strip()]
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".") if r.strip()]
            
            if not products or not reactants:
                return False
                
            # Look for aromatic substitution patterns
            return self._detect_snar_pattern(reactants, products)
            
        except Exception:
            return False
    
    def _detect_snar_pattern(self, reactants, products) -> bool:
        """
        Detects nucleophilic aromatic substitution pattern.
        """
        # Common leaving groups in SNAr
        leaving_group_patterns = [
            "[cH0:1][F,Cl,Br,I]",  # Aryl halides
            "[cH0:1][N+](=O)[O-]",  # Aryl nitro compounds
            "[cH0:1]S(=O)(=O)[CH3]",  # Aryl mesylates
        ]
        
        # Nucleophile patterns that form new bonds to aromatic carbons
        nucleophile_patterns = [
            "[NH2,NH1,NH0]",  # Amines
            "[OH1,OH0]",      # Alcohols/phenols
            "[SH1,SH0]",      # Thiols/thiolates
            "[CH2][O][CH3]",  # MOM ether (specific to the rationale)
        ]
        
        # Check for electron-withdrawing groups on aromatic ring
        ewg_patterns = [
            "c[N+](=O)[O-]",     # Nitro
            "c[C](=O)",          # Carbonyl
            "c[C]#[N]",          # Cyano
            "c[C](F)(F)F",       # Trifluoromethyl
        ]
        
        for reactant in reactants:
            if not reactant:
                continue
                
            # Check if reactant has aromatic ring with leaving group
            has_leaving_group = any(
                reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern))
                for pattern in leaving_group_patterns
            )
            
            # Check for electron-withdrawing groups
            has_ewg = any(
                reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern))
                for pattern in ewg_patterns
            )
            
            if has_leaving_group and has_ewg:
                # Check if any product shows nucleophilic substitution
                for product in products:
                    if not product:
                        continue
                        
                    # Look for new C-N, C-O, or C-S bonds to aromatic carbon
                    new_bond_patterns = [
                        "c[NH2,NH1,NH0]",  # C-N bond
                        "c[OH1,OH0]",      # C-O bond  
                        "c[SH1,SH0]",      # C-S bond
                        "c[O][CH2][O][CH3]",  # MOM ether specifically
                    ]
                    
                    has_new_bond = any(
                        product.HasSubstructMatch(Chem.MolFromSmarts(pattern))
                        for pattern in new_bond_patterns
                    )
                    
                    if has_new_bond:
                        return True
        
        return False
