"""Generated evaluation code for: TBS protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TBSCyclingStrategy(MultiRxnCondBase):
    """
    Detects TBS protecting group cycling strategy where TBS protection is applied 
    and then removed in consecutive steps for the same alcohol functionality.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "TBS")
        self.strategy = config.get("strategy", "cycling")
        
        # TBS group SMARTS pattern
        self.tbs_pattern = Chem.MolFromSmarts("[Si](C)(C)(C(C)(C)C)")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track TBS protection/deprotection events with atom mapping
        protection_events = []
        deprotection_events = []
        
        for i, rxn in enumerate(reactions):
            rxn_smiles = rxn["metadata"]["mapped_reaction_smiles"]
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            
            prod_mol = Chem.MolFromSmiles(prod_smiles)
            react_mols = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
            
            # Check for TBS protection (TBS appears in product but not reactants)
            if self.has_tbs_group(prod_mol):
                tbs_atoms_prod = self.get_tbs_connected_atoms(prod_mol)
                
                has_tbs_in_reactants = any(self.has_tbs_group(r) for r in react_mols if r is not None)
                
                if not has_tbs_in_reactants and tbs_atoms_prod:
                    # This is a protection event
                    for atom_map in tbs_atoms_prod:
                        protection_events.append((i, atom_map))
            
            # Check for TBS deprotection (TBS in reactants but not product)
            reactant_tbs_atoms = []
            for r_mol in react_mols:
                if r_mol is not None and self.has_tbs_group(r_mol):
                    reactant_tbs_atoms.extend(self.get_tbs_connected_atoms(r_mol))
            
            if reactant_tbs_atoms and not self.has_tbs_group(prod_mol):
                # This is a deprotection event
                for atom_map in reactant_tbs_atoms:
                    deprotection_events.append((i, atom_map))
        
        # Check for cycling: same atom is protected and then deprotected
        cycling_detected = self.detect_cycling(protection_events, deprotection_events)
        
        return cycling_detected, len(reactions)
    
    def has_tbs_group(self, mol):
        """Check if molecule contains TBS group"""
        if mol is None:
            return False
        return mol.HasSubstructMatch(self.tbs_pattern)
    
    def get_tbs_connected_atoms(self, mol):
        """Get atom map numbers of atoms connected to TBS silicon"""
        if mol is None:
            return []
        
        connected_atoms = []
        matches = mol.GetSubstructMatches(self.tbs_pattern)
        
        for match in matches:
            si_idx = match[0]  # Silicon is first atom in pattern
            si_atom = mol.GetAtomWithIdx(si_idx)
            
            # Find oxygen connected to silicon (TBS-O-R pattern)
            for neighbor in si_atom.GetNeighbors():
                if neighbor.GetSymbol() == 'O':
                    # Find carbon connected to this oxygen
                    for o_neighbor in neighbor.GetNeighbors():
                        if o_neighbor.GetSymbol() == 'C' and o_neighbor.GetIdx() != si_idx:
                            atom_map = o_neighbor.GetAtomMapNum()
                            if atom_map > 0:
                                connected_atoms.append(atom_map)
        
        return connected_atoms
    
    def detect_cycling(self, protection_events, deprotection_events):
        """Detect if the same atom undergoes protection followed by deprotection"""
        for prot_step, prot_atom in protection_events:
            for deprot_step, deprot_atom in deprotection_events:
                # Same atom and deprotection happens after protection
                if prot_atom == deprot_atom and deprot_step > prot_step:
                    return True
        return False
    
    def route_scoring(self, x):
        """Score based on presence of cycling strategy"""
        if x < 0:
            return 0  # No cycling detected
        else:
            return 10  # Cycling strategy detected
