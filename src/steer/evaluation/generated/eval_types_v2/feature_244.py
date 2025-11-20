"""Generated evaluation code for: Late stage ether formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageEtherFormation(BaseScoring):
    """
    Evaluates whether Williamson ether synthesis (C-O bond formation) occurs 
    in the late stages of the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config["parameters"].get("timing", "late")
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score.
        For late-stage preference: higher depth fraction = better score
        """
        if x < 0:
            return 0  # Ether formation doesn't occur
        
        if self.timing_preference == "late":
            return x * 10  # Late stage (high depth fraction) gets high score
        else:
            return (1 - x) * 10  # Early stage preference
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents Williamson ether synthesis.
        Detects C-O bond formation between an alkoxide and alkyl halide/tosylate.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, product_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if we're forming a new C-O bond
            if self._is_williamson_ether_synthesis(reactants, product):
                return True
                
        except Exception:
            pass
            
        return False
    
    def _is_williamson_ether_synthesis(self, reactants, product):
        """
        Detect Williamson ether synthesis pattern:
        - Formation of new C-O-C linkage
        - Typical reactants: alkoxide + alkyl halide/tosylate
        """
        # Look for ether formation patterns in reactants vs product
        ether_pattern = Chem.MolFromSmarts("[C]-O-[C]")  # Simple ether linkage
        alkoxide_pattern = Chem.MolFromSmarts("[C]-[O-]")  # Alkoxide nucleophile
        alkyl_halide_pattern = Chem.MolFromSmarts("[C]-[Cl,Br,I]")  # Alkyl halide
        tosylate_pattern = Chem.MolFromSmarts("[C]-OS(=O)(=O)[c]")  # Tosylate leaving group
        
        if not product.HasSubstructMatch(ether_pattern):
            return False
            
        # Check if reactants contain typical Williamson ether precursors
        has_alkoxide = False
        has_electrophile = False
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(alkoxide_pattern):
                has_alkoxide = True
            if (reactant.HasSubstructMatch(alkyl_halide_pattern) or 
                reactant.HasSubstructMatch(tosylate_pattern)):
                has_electrophile = True
        
        # Additional check: count C-O bonds in reactants vs product
        reactant_co_bonds = sum(self._count_co_bonds(r) for r in reactants)
        product_co_bonds = self._count_co_bonds(product)
        
        # Williamson ether synthesis should increase C-O bond count
        co_bond_formed = product_co_bonds > reactant_co_bonds
        
        return (has_alkoxide and has_electrophile) or co_bond_formed
    
    def _count_co_bonds(self, mol):
        """Count C-O bonds in a molecule"""
        count = 0
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            if ((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'O') or
                (atom1.GetSymbol() == 'O' and atom2.GetSymbol() == 'C')):
                count += 1
        return count
