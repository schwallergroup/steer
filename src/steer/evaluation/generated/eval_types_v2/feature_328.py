"""Generated evaluation code for: Late stage ether formation via Williamson synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageWilliamsonEther(BaseScoring):
    """
    Evaluates late-stage ether formation via Williamson synthesis.
    Checks for C-O bond formation through alkylation reactions at late stages.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config["parameters"].get("depth_threshold", 3)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Williamson ether synthesis doesn't occur
        else:
            # Late-stage is better, penalize early reactions
            if x <= 1.0 / self.depth_threshold:
                return 1.0  # Very late stage - good score
            else:
                return max(0, 1.0 - x)  # Earlier stages get lower scores
    
    def hit_condition(self, d) -> bool:
        """
        Detects Williamson ether synthesis by looking for:
        1. C-O bond formation 
        2. Alkyl halide + alkoxide pattern
        3. Dibromoethane alkylation pattern
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Check for C-O ether bond formation
            if not self._has_ether_formation(reactants, product):
                return False
                
            # Check for Williamson synthesis patterns
            return self._is_williamson_pattern(reactants)
            
        except Exception:
            return False
    
    def _has_ether_formation(self, reactants, product) -> bool:
        """Check if new C-O ether bonds are formed"""
        # Count C-O bonds in product vs reactants
        product_co_bonds = self._count_co_ether_bonds(product)
        reactant_co_bonds = sum(self._count_co_ether_bonds(mol) for mol in reactants)
        
        return product_co_bonds > reactant_co_bonds
    
    def _count_co_ether_bonds(self, mol) -> int:
        """Count C-O ether bonds (not C=O or C-OH)"""
        if not mol:
            return 0
            
        count = 0
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            
            # Check for C-O single bond
            if (bond.GetBondType() == Chem.BondType.SINGLE and 
                ((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'O') or
                 (atom1.GetSymbol() == 'O' and atom2.GetSymbol() == 'C'))):
                
                # Exclude alcohols (O with H) and carbonyls
                oxygen = atom2 if atom2.GetSymbol() == 'O' else atom1
                carbon = atom1 if atom2.GetSymbol() == 'O' else atom2
                
                # Check if oxygen is not bonded to hydrogen (exclude alcohols)
                has_oh = any(neighbor.GetSymbol() == 'H' for neighbor in oxygen.GetNeighbors())
                
                # Check if carbon is not double bonded to oxygen (exclude carbonyls)
                has_co_double = any(b.GetBondType() == Chem.BondType.DOUBLE and 
                                   b.GetOtherAtom(carbon).GetSymbol() == 'O' 
                                   for b in carbon.GetBonds())
                
                if not has_oh and not has_co_double:
                    count += 1
                    
        return count
    
    def _is_williamson_pattern(self, reactants) -> bool:
        """Check for Williamson synthesis reactant patterns"""
        # Look for alkyl halide patterns
        halide_patterns = [
            "[C][Br]",  # Alkyl bromide
            "[C][I]",   # Alkyl iodide  
            "[C][Cl]",  # Alkyl chloride
            "BrCCBr",   # Dibromoethane
            "BrCC[Br]"  # Alternative dibromoethane pattern
        ]
        
        # Look for alkoxide/alcohol patterns
        alkoxide_patterns = [
            "[C][O-]",  # Alkoxide
            "[C][OH]",  # Alcohol (can be deprotonated)
            "c[OH]",    # Phenol
            "c[O-]"     # Phenoxide
        ]
        
        has_halide = False
        has_nucleophile = False
        
        for reactant in reactants:
            # Check for halide patterns
            for pattern in halide_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_halide = True
                    break
                    
            # Check for nucleophile patterns
            for pattern in alkoxide_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_nucleophile = True
                    break
        
        return has_halide and has_nucleophile
