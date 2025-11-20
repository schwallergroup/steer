"""Generated evaluation code for: Reductive cyclization for lactam formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ReductiveCyclizationLactam(BaseScoring):
    """
    Evaluates synthesis routes for the presence of reductive cyclization reactions
    that form lactams from nitro-ester precursors. This transformation involves
    simultaneous reduction of a nitro group and cyclization to form a lactam ring.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to 0-10 score, rewarding earlier occurrence"""
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Earlier occurrence is better (lower x values get higher scores)
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """Check if a reaction represents reductive cyclization to lactam"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactant_smiles, product_smiles = mapped_rxn.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(smi.strip()) for smi in product_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactant_smiles.split(".")]
            
            # Filter out None molecules
            products = [mol for mol in products if mol is not None]
            reactants = [mol for mol in reactants if mol is not None]
            
            if not products or not reactants:
                return False
            
            # Check for lactam formation and nitro/ester consumption
            return (self._has_lactam_formation(reactants, products) and 
                    self._has_nitro_reduction(reactants, products) and
                    self._has_ester_involvement(reactants))
        
        except Exception:
            return False
    
    def _has_lactam_formation(self, reactants, products) -> bool:
        """Check if lactam rings are formed in the reaction"""
        # Lactam patterns (4, 5, 6, 7-membered rings)
        lactam_patterns = [
            "C1CNC(=O)C1",      # 5-membered lactam (pyrrolidin-2-one)
            "C1CCNC(=O)C1",     # 6-membered lactam (piperidin-2-one)  
            "C1CCCNC(=O)C1",    # 7-membered lactam (azepan-2-one)
            "C1NC(=O)CC1",      # Alternative 5-membered
            "C1CCC(=O)NC1",     # 6-membered (different position)
            "N1C(=O)CCCC1",     # 6-membered lactam
            "N1C(=O)CCC1"       # 5-membered lactam
        ]
        
        # Check if products have more lactam substructures than reactants
        reactant_lactams = sum(self._count_substructures(mol, lactam_patterns) for mol in reactants)
        product_lactams = sum(self._count_substructures(mol, lactam_patterns) for mol in products)
        
        return product_lactams > reactant_lactams
    
    def _has_nitro_reduction(self, reactants, products) -> bool:
        """Check if nitro groups are reduced (nitro count decreases)"""
        nitro_pattern = "[N+](=O)[O-]"
        
        reactant_nitros = sum(self._count_substructures(mol, [nitro_pattern]) for mol in reactants)
        product_nitros = sum(self._count_substructures(mol, [nitro_pattern]) for mol in products)
        
        return reactant_nitros > product_nitros and reactant_nitros > 0
    
    def _has_ester_involvement(self, reactants) -> bool:
        """Check if ester groups are present in reactants"""
        ester_patterns = [
            "C(=O)OC",          # Generic ester
            "[C](=O)O[C]",      # Ester with explicit carbons
            "C(=O)O[!H]"        # Ester (carbonyl-oxygen-non-hydrogen)
        ]
        
        return any(self._count_substructures(mol, ester_patterns) > 0 for mol in reactants)
    
    def _count_substructures(self, mol, patterns) -> int:
        """Count total occurrences of given SMARTS patterns in molecule"""
        if mol is None:
            return 0
        
        count = 0
        for pattern in patterns:
            try:
                patt_mol = Chem.MolFromSmarts(pattern)
                if patt_mol is not None:
                    matches = mol.GetSubstructMatches(patt_mol)
                    count += len(matches)
            except Exception:
                continue
        
        return count
