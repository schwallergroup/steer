"""Generated evaluation code for: Multiple protecting group cycling on same position"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Detects multiple protecting group cycling on the same position.
    Checks if a route deprotects one group then immediately reprotects 
    the same position with a different protecting group.
    """
    
    def __init__(self, config):
        self.position = config["parameters"]["position"]
        self.groups = config["parameters"]["groups"]
        self.cycling = config["parameters"]["cycling"]
        
        # Define protecting group patterns
        self.protecting_patterns = {
            "Boc": "[NH1,NH2][C](=O)O[C](C)(C)C",
            "phthalimide": "[NH1]C1=CC=C2C(=O)[NH1]C(=O)C2=C1",
            "Cbz": "[NH1,NH2][C](=O)OCC1=CC=CC=C1",
            "Fmoc": "[NH1,NH2][C](=O)OCC1C2=CC=CC=C2C2=CC=CC=C12"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if protecting group cycling occurs in the route."""
        reactions = self.get_rxns(d)
        
        if len(reactions) < 2:
            return False, len(reactions)
        
        # Track protection/deprotection events by atom mapping
        protection_events = []
        
        for i, rxn_data in enumerate(reactions):
            rxn_smiles = rxn_data["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                continue
                
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Check for deprotection (protected -> free amine)
            deprotection = self.detect_deprotection(reactants, products)
            if deprotection:
                protection_events.append(("deprotect", deprotection["group"], 
                                        deprotection["atom_map"], i))
            
            # Check for protection (free amine -> protected)
            protection = self.detect_protection(reactants, products)
            if protection:
                protection_events.append(("protect", protection["group"], 
                                        protection["atom_map"], i))
        
        # Look for cycling pattern: deprotect group A, then protect with group B
        cycling_detected = self.detect_cycling_pattern(protection_events)
        
        return cycling_detected == self.cycling, len(reactions)
    
    def detect_deprotection(self, reactants, products):
        """Detect if a protecting group is removed."""
        try:
            react_mol = Chem.MolFromSmiles(reactants)
            prod_mol = Chem.MolFromSmiles(products)
            
            if not react_mol or not prod_mol:
                return None
            
            # Check each protecting group pattern
            for group_name in self.groups:
                if group_name not in self.protecting_patterns:
                    continue
                    
                pattern = Chem.MolFromSmarts(self.protecting_patterns[group_name])
                if not pattern:
                    continue
                
                # Present in reactants but not products
                if react_mol.HasSubstructMatch(pattern):
                    matches_react = react_mol.GetSubstructMatches(pattern)
                    matches_prod = prod_mol.GetSubstructMatches(pattern) if prod_mol.HasSubstructMatch(pattern) else []
                    
                    # Find nitrogen atom that lost protection
                    for match in matches_react:
                        n_idx = match[0]  # First atom in pattern is nitrogen
                        n_atom = react_mol.GetAtomWithIdx(n_idx)
                        atom_map = n_atom.GetAtomMapNum()
                        
                        if atom_map > 0:
                            # Check if this nitrogen is now free in products
                            prod_n = None
                            for atom in prod_mol.GetAtoms():
                                if atom.GetAtomMapNum() == atom_map:
                                    prod_n = atom
                                    break
                            
                            if prod_n and self.is_free_amine(prod_mol, prod_n.GetIdx()):
                                return {"group": group_name, "atom_map": atom_map}
                                
        except Exception:
            pass
        
        return None
    
    def detect_protection(self, reactants, products):
        """Detect if a free amine gets protected."""
        try:
            react_mol = Chem.MolFromSmiles(reactants)
            prod_mol = Chem.MolFromSmiles(products)
            
            if not react_mol or not prod_mol:
                return None
            
            # Find free amines in reactants
            free_amines = []
            for atom in react_mol.GetAtoms():
                if atom.GetSymbol() == 'N' and atom.GetAtomMapNum() > 0:
                    if self.is_free_amine(react_mol, atom.GetIdx()):
                        free_amines.append(atom.GetAtomMapNum())
            
            # Check if any became protected in products
            for group_name in self.groups:
                if group_name not in self.protecting_patterns:
                    continue
                    
                pattern = Chem.MolFromSmarts(self.protecting_patterns[group_name])
                if not pattern:
                    continue
                
                if prod_mol.HasSubstructMatch(pattern):
                    matches = prod_mol.GetSubstructMatches(pattern)
                    for match in matches:
                        n_idx = match[0]
                        n_atom = prod_mol.GetAtomWithIdx(n_idx)
                        atom_map = n_atom.GetAtomMapNum()
                        
                        if atom_map in free_amines:
                            return {"group": group_name, "atom_map": atom_map}
                            
        except Exception:
            pass
        
        return None
    
    def is_free_amine(self, mol, n_idx):
        """Check if nitrogen is a free amine (primary or secondary)."""
        try:
            atom = mol.GetAtomWithIdx(n_idx)
            if atom.GetSymbol() != 'N':
                return False
            
            # Count non-hydrogen neighbors
            heavy_neighbors = 0
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() != 'H':
                    heavy_neighbors += 1
            
            # Free primary amine (NH2): 0-1 heavy neighbors
            # Free secondary amine (NHR): 1-2 heavy neighbors
            return heavy_neighbors <= 2 and atom.GetTotalNumHs() > 0
            
        except Exception:
            return False
    
    def detect_cycling_pattern(self, events):
        """Detect if there's a deprotect->protect cycle on same position."""
        if len(events) < 2:
            return False
        
        # Sort events by reaction index
        events.sort(key=lambda x: x[3])
        
        # Look for consecutive or nearby deprotect->protect on same atom
        for i in range(len(events) - 1):
            event1 = events[i]
            
            # Find subsequent events on same atom position
            for j in range(i + 1, min(i + 3, len(events))):  # Check next 2 reactions
                event2 = events[j]
                
                # Same atom map number (same position)
                if event1[2] == event2[2]:
                    # Deprotection followed by protection with different group
                    if (event1[0] == "deprotect" and event2[0] == "protect" and 
                        event1[1] != even
