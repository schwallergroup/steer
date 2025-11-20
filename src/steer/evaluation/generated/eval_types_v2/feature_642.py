"""Generated evaluation code for: Late stage Buchwald-Hartwig amination coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmination(BaseScoring):
    """
    Evaluates whether Buchwald-Hartwig amination coupling occurs in late stage synthesis.
    
    Detects the formation of C-N bonds through palladium-catalyzed cross-coupling
    between aryl halides/pseudohalides and amines within the final steps of synthesis.
    """
    
    def __init__(self, config: Dict):
        self.stage_cutoff = config.get("stage_cutoff", 2)
        self.timing = config.get("timing", "late")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        
        if self.timing == "late":
            # Reward reactions closer to the end (higher depth fraction)
            return 10 * x
        else:
            # For early timing, reward lower depth fractions
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents a Buchwald-Hartwig amination"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for C-N bond formation
            if not self._has_cn_bond_formation(reactants, product):
                return False
            
            # Check for Buchwald-Hartwig pattern
            return self._is_buchwald_hartwig_pattern(reactants, product)
            
        except Exception:
            return False
    
    def _has_cn_bond_formation(self, reactants, product) -> bool:
        """Check if C-N bonds are formed in the reaction"""
        # Count C-N bonds in reactants vs product
        reactant_cn_bonds = sum(self._count_cn_bonds(mol) for mol in reactants)
        product_cn_bonds = self._count_cn_bonds(product)
        
        return product_cn_bonds > reactant_cn_bonds
    
    def _count_cn_bonds(self, mol) -> int:
        """Count C-N bonds in a molecule"""
        if not mol:
            return 0
        
        count = 0
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            if ((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'N') or
                (atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'C')):
                count += 1
        return count
    
    def _is_buchwald_hartwig_pattern(self, reactants, product) -> bool:
        """Check for characteristic Buchwald-Hartwig coupling patterns"""
        # Look for aryl halide/pseudohalide pattern
        aryl_halide_patterns = [
            "[cH0:1][Cl,Br,I,F]",  # Aryl halides
            "[cH0:1][O][S](=O)(=O)[C,c]",  # Aryl tosylates/mesylates
            "[cH0:1][O][S](=O)(=O)C(F)(F)F"  # Aryl triflates
        ]
        
        # Look for amine patterns
        amine_patterns = [
            "[NH2]",  # Primary amines
            "[NH1]",  # Secondary amines
            "N1CCCCC1",  # Piperidine
            "N1CCCC1",   # Pyrrolidine
            "c1ccc(N)cc1"  # Anilines
        ]
        
        # Check if reactants contain these patterns
        has_aryl_halide = False
        has_amine = False
        
        for reactant in reactants:
            # Check for aryl halide patterns
            for pattern in aryl_halide_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_aryl_halide = True
                    break
            
            # Check for amine patterns
            for pattern in amine_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_amine = True
                    break
        
        # Also check for the formation of aryl-amine bond in product
        aryl_amine_product = "[c:1][NH1,NH2]"
        has_aryl_amine_product = product.HasSubstructMatch(Chem.MolFromSmarts(aryl_amine_product))
        
        return has_aryl_halide and has_amine and has_aryl_amine_product
