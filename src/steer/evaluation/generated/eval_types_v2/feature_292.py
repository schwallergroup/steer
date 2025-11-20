"""Generated evaluation code for: Sequential Boc-SEM protecting group swap strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocSemProtectingGroupSwap(MultiRxnCondBase):
    """
    Evaluates synthesis routes for sequential Boc-SEM protecting group swap strategy.
    Checks if Boc protection is first installed on imidazole nitrogen, then removed 
    and replaced with SEM protection on the same nitrogen atom.
    """
    
    def __init__(self, config):
        self.initial_group = config.get("initial_group", "Boc")
        self.final_group = config.get("final_group", "SEM")
        self.substrate = config.get("substrate", "imidazole_nitrogen")
        self.sequential_swap = config.get("sequential_swap", True)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track sequence of protecting group operations
        boc_install_found = False
        boc_remove_found = False
        sem_install_found = False
        imidazole_map_num = None
        
        # Process reactions in chronological order (reverse of synthesis tree)
        for rxn in reversed(reactions):
            rxn_smiles = rxn["metadata"]["mapped_reaction_smiles"]
            
            # Check for Boc installation on imidazole nitrogen
            if not boc_install_found:
                boc_installed, map_num = self.detect_boc_installation(rxn_smiles)
                if boc_installed:
                    boc_install_found = True
                    imidazole_map_num = map_num
                    continue
            
            # Check for Boc removal (must be on same nitrogen)
            if boc_install_found and not boc_remove_found and imidazole_map_num:
                if self.detect_boc_removal(rxn_smiles, imidazole_map_num):
                    boc_remove_found = True
                    continue
            
            # Check for SEM installation (must be on same nitrogen after Boc removal)
            if boc_remove_found and not sem_install_found and imidazole_map_num:
                if self.detect_sem_installation(rxn_smiles, imidazole_map_num):
                    sem_install_found = True
                    break
        
        # Sequential swap requires all three steps in correct order
        condition = boc_install_found and boc_remove_found and sem_install_found
        return condition, len(reactions)
    
    def detect_boc_installation(self, rxn_smiles) -> Tuple[bool, int]:
        """Detect Boc protection installation on imidazole nitrogen"""
        prod_smiles, react_smiles = rxn_smiles.split(">>")
        prod = Chem.MolFromSmiles(prod_smiles)
        reactants = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
        
        # Pattern for N-Boc imidazole
        boc_imidazole_pattern = Chem.MolFromSmarts("[nH0:1]1c[nH]cc1C(=O)OC(C)(C)C")
        # Pattern for free imidazole nitrogen
        free_imidazole_pattern = Chem.MolFromSmarts("[nH0:1]1c[nH]cc1")
        
        if not prod.HasSubstructMatch(boc_imidazole_pattern):
            return False, None
            
        # Find the nitrogen that got protected
        match = prod.GetSubstructMatch(boc_imidazole_pattern)
        if not match:
            return False, None
            
        n_atom = prod.GetAtomWithIdx(match[0])
        map_num = n_atom.GetAtomMapNum()
        
        # Verify reactant had free imidazole
        for reactant in reactants:
            if reactant.HasSubstructMatch(free_imidazole_pattern):
                react_match = reactant.GetSubstructMatch(free_imidazole_pattern)
                if react_match:
                    react_n = reactant.GetAtomWithIdx(react_match[0])
                    if react_n.GetAtomMapNum() == map_num:
                        return True, map_num
                        
        return False, None
    
    def detect_boc_removal(self, rxn_smiles, target_map_num) -> bool:
        """Detect Boc deprotection from specific nitrogen"""
        prod_smiles, react_smiles = rxn_smiles.split(">>")
        prod = Chem.MolFromSmiles(prod_smiles)
        reactants = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
        
        # Pattern for N-Boc imidazole
        boc_imidazole_pattern = Chem.MolFromSmarts("[nH0:1]1c[nH]cc1C(=O)OC(C)(C)C")
        # Pattern for free imidazole nitrogen
        free_imidazole_pattern = Chem.MolFromSmarts("[nH0:1]1c[nH]cc1")
        
        # Product should have free imidazole at target position
        if not prod.HasSubstructMatch(free_imidazole_pattern):
            return False
            
        prod_match = prod.GetSubstructMatch(free_imidazole_pattern)
        prod_n = prod.GetAtomWithIdx(prod_match[0])
        if prod_n.GetAtomMapNum() != target_map_num:
            return False
        
        # Reactant should have Boc-protected imidazole at same position
        for reactant in reactants:
            if reactant.HasSubstructMatch(boc_imidazole_pattern):
                react_match = reactant.GetSubstructMatch(boc_imidazole_pattern)
                react_n = reactant.GetAtomWithIdx(react_match[0])
                if react_n.GetAtomMapNum() == target_map_num:
                    return True
                    
        return False
    
    def detect_sem_installation(self, rxn_smiles, target_map_num) -> bool:
        """Detect SEM protection installation on specific nitrogen"""
        prod_smiles, react_smiles = rxn_smiles.split(">>")
        prod = Chem.MolFromSmiles(prod_smiles)
        reactants = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
        
        # Pattern for N-SEM imidazole (2-(trimethylsilyl)ethoxymethyl)
        sem_imidazole_pattern = Chem.MolFromSmarts("[nH0:1]1c[nH]cc1COCC[Si](C)(C)C")
        # Pattern for free imidazole nitrogen
        free_imidazole_pattern = Chem.MolFromSmarts("[nH0:1]1c[nH]cc1")
        
        if not prod.HasSubstructMatch(sem_imidazole_pattern):
            return False
            
        prod_match = prod.GetSubstructMatch(sem_imidazole_pattern)
        prod_n = prod.GetAtomWithIdx(prod_match[0])
        if prod_n.GetAtomMapNum() != target_map_num:
            return False
        
        # Reactant should have free imidazole at same position
        for reactant in reactants:
            if reactant.HasSubstructMatch(free_imidazole_pattern):
                react_match = reactant.GetSubstructMatch(free_imidazole_pattern)
                react_n = reactant.GetAtomWithIdx(react_match[0])
                if react_n.GetAtomMapNum() == target_map_num:
                    return Tru
