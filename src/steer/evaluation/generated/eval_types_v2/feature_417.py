"""Generated evaluation code for: Late stage Williamson ether coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageWilliamsonEther(BaseScoring):
    """
    Evaluates whether a Williamson ether synthesis (C-O ether bond formation)
    occurs at late stage in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.0)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Williamson ether coupling doesn't happen
        else:
            # Late-stage (lower depth fraction) is better for ether formation
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Detects Williamson ether synthesis by identifying C-O ether bond formation
        between an alkyl halide/tosylate and a phenoxide/alkoxide.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Look for C-O ether bonds formed in product that weren't in reactants
            product_ethers = self._find_ether_bonds(product)
            reactant_ethers = set()
            for reactant in reactants:
                reactant_ethers.update(self._find_ether_bonds(reactant))
            
            # Check if new ether bonds are formed
            new_ethers = product_ethers - reactant_ethers
            if not new_ethers:
                return False
            
            # Verify it's a Williamson-type mechanism:
            # 1. One reactant should have a leaving group (halide, tosylate)
            # 2. One reactant should be/form an alkoxide or phenoxide
            return self._is_williamson_mechanism(reactants, new_ethers)
            
        except Exception:
            return False
    
    def _find_ether_bonds(self, mol) -> set:
        """Find C-O-C ether bonds in molecule using atom map numbers."""
        ether_bonds = set()
        
        # SMARTS pattern for ether: [C]-[O]-[C] (excluding alcohols, acids, esters)
        ether_pattern = Chem.MolFromSmarts("[C:1]-[O:2]-[C:3]")
        matches = mol.GetSubstructMatches(ether_pattern)
        
        for match in matches:
            c1_idx, o_idx, c2_idx = match
            # Get atom map numbers if available
            c1_map = mol.GetAtomWithIdx(c1_idx).GetAtomMapNum()
            o_map = mol.GetAtomWithIdx(o_idx).GetAtomMapNum()
            c2_map = mol.GetAtomWithIdx(c2_idx).GetAtomMapNum()
            
            # Skip if this oxygen is part of carbonyl, carboxyl, etc.
            o_atom = mol.GetAtomWithIdx(o_idx)
            if o_atom.GetTotalValence() != 2 or o_atom.GetTotalNumHs() > 0:
                continue
                
            if c1_map and o_map and c2_map:
                ether_bonds.add(tuple(sorted([c1_map, o_map, c2_map])))
        
        return ether_bonds
    
    def _is_williamson_mechanism(self, reactants, new_ethers) -> bool:
        """Check if the reaction follows Williamson ether synthesis pattern."""
        has_alkyl_halide = False
        has_nucleophile = False
        
        # SMARTS patterns for common leaving groups in Williamson synthesis
        alkyl_halide_patterns = [
            "[C][Cl,Br,I]",  # Alkyl halides
            "[C]OS(=O)(=O)[c]",  # Tosylates
            "[C]OS(=O)(=O)C"  # Mesylates
        ]
        
        # SMARTS patterns for nucleophiles (phenols, alcohols that can be deprotonated)
        nucleophile_patterns = [
            "[OH][c]",  # Phenols
            "[OH][C]",  # Alcohols (can form alkoxides)
            "[O-]",     # Already deprotonated
        ]
        
        for reactant in reactants:
            # Check for leaving groups
            for pattern_smarts in alkyl_halide_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and reactant.HasSubstructMatch(pattern):
                    has_alkyl_halide = True
                    break
            
            # Check for nucleophile precursors
            for pattern_smarts in nucleophile_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and reactant.HasSubstructMatch(pattern):
                    has_nucleophile = True
                    break
        
        return has_alkyl_halide and has_nucleophile
