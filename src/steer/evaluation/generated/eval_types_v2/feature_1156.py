"""Generated evaluation code for: Sequential Williamson ether synthesis strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialWilliamsonEther(MultiRxnCondBase):
    """
    Evaluates routes containing sequential Williamson ether synthesis reactions.
    Checks for two consecutive Williamson ether formations used to build diaryl 
    ether linkages via an ethylene bridge.
    """
    
    def __init__(self, config):
        self.required_count = config.get("count", 2)
        self.sequential = config.get("sequential", True)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        williamson_reactions = []
        
        # Find all Williamson ether reactions and their positions
        for i, rxn in enumerate(reactions):
            if self.detect_williamson_ether(rxn):
                williamson_reactions.append(i)
        
        # Check if we have the required count
        if len(williamson_reactions) < self.required_count:
            return False, len(reactions)
        
        # If sequential is required, check for consecutive reactions
        if self.sequential and self.required_count == 2:
            for i in range(len(williamson_reactions) - 1):
                if williamson_reactions[i+1] - williamson_reactions[i] == 1:
                    return True, len(reactions)
            return False, len(reactions)
        
        # If not sequential, just check count
        return len(williamson_reactions) >= self.required_count, len(reactions)
    
    def detect_williamson_ether(self, rxn):
        """
        Detects Williamson ether synthesis by looking for:
        1. Formation of C-O-C ether bond
        2. Typical nucleophilic substitution pattern (alkoxide + alkyl halide)
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Look for alkyl halide pattern in reactants
            halide_pattern = Chem.MolFromSmarts("[C,c][Cl,Br,I]")
            has_halide = any(mol.HasSubstructMatch(halide_pattern) for mol in reactants)
            
            # Look for phenoxide or alkoxide pattern in reactants
            phenoxide_pattern = Chem.MolFromSmarts("[c,C][O-]")
            alkoxide_pattern = Chem.MolFromSmarts("[C][O-]")
            has_alkoxide = any(mol.HasSubstructMatch(phenoxide_pattern) or 
                             mol.HasSubstructMatch(alkoxide_pattern) for mol in reactants)
            
            # Look for ether formation in products
            ether_pattern = Chem.MolFromSmarts("[C,c]O[C,c]")
            has_ether_product = any(mol.HasSubstructMatch(ether_pattern) for mol in products)
            
            # Check for halide leaving group in products
            halide_ion_pattern = Chem.MolFromSmarts("[Cl-,Br-,I-]")
            has_halide_product = any(mol.HasSubstructMatch(halide_ion_pattern) for mol in products)
            
            # Williamson ether synthesis typically involves:
            # - Alkyl halide + alkoxide -> ether + halide ion
            return (has_halide and has_alkoxide and has_ether_product) or \
                   (has_halide and has_ether_product and has_halide_product)
                   
        except Exception:
            return False
