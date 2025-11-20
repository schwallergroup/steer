"""Generated evaluation code for: Convergent assembly via alkyne-azomethine cycloaddition"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AlkyneAzomethineCycloaddition(BaseScoring):
    """
    Evaluates convergent assembly via alkyne-azomethine cycloaddition.
    Checks for cycloaddition reactions that form C-N bonds between alkyne-containing
    fragments and azomethine/imine-containing fragments.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.3)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cycloaddition doesn't happen
        else:
            if self.condition_type == "bool":
                return 1  # Condition met
            else:
                # Earlier cycloaddition (lower depth) is better for convergent synthesis
                return max(0, 1 - abs(x - self.target_depth))
    
    def hit_condition(self, d):
        """Check if reaction is an alkyne-azomethine cycloaddition"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product or len(reactants) < 2:
                return False
                
            # Check if we have the right number of fragments
            if len(reactants) != self.fragment_count:
                return False
                
            # Check for alkyne and azomethine/imine patterns in reactants
            alkyne_pattern = Chem.MolFromSmarts("[C]#[C]")
            azomethine_pattern = Chem.MolFromSmarts("[C]=[N]")  # C=N bond (azomethine/imine)
            pyridine_pattern = Chem.MolFromSmarts("c1ccncc1")  # pyrimidine-like N-heterocycle
            
            has_alkyne = False
            has_azomethine = False
            
            for reactant in reactants:
                if reactant.HasSubstructMatch(alkyne_pattern):
                    has_alkyne = True
                if reactant.HasSubstructMatch(azomethine_pattern) or reactant.HasSubstructMatch(pyridine_pattern):
                    has_azomethine = True
                    
            if not (has_alkyne and has_azomethine):
                return False
                
            # Check if cycloaddition occurred by looking for new C-N bonds formed
            return self._check_cycloaddition_occurred(product, reactants)
            
        except Exception:
            return False
    
    def _check_cycloaddition_occurred(self, product, reactants):
        """Check if cycloaddition reaction formed new C-N bonds"""
        try:
            # Get atom map numbers for tracking atoms through reaction
            product_atoms = {atom.GetAtomMapNum(): atom for atom in product.GetAtoms() if atom.GetAtomMapNum() > 0}
            
            reactant_atoms = {}
            for reactant in reactants:
                for atom in reactant.GetAtoms():
                    if atom.GetAtomMapNum() > 0:
                        reactant_atoms[atom.GetAtomMapNum()] = atom
                        
            # Look for new C-N bonds in product that weren't in reactants
            new_cn_bonds = 0
            
            for bond in product.GetBonds():
                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()
                
                begin_map = begin_atom.GetAtomMapNum()
                end_map = end_atom.GetAtomMapNum()
                
                if begin_map == 0 or end_map == 0:
                    continue
                    
                # Check if this is a C-N bond
                atom_symbols = sorted([begin_atom.GetSymbol(), end_atom.GetSymbol()])
                if atom_symbols != ['C', 'N']:
                    continue
                    
                # Check if these atoms were in different reactants
                begin_reactant = None
                end_reactant = None
                
                for i, reactant in enumerate(reactants):
                    reactant_maps = [a.GetAtomMapNum() for a in reactant.GetAtoms()]
                    if begin_map in reactant_maps:
                        begin_reactant = i
                    if end_map in reactant_maps:
                        end_reactant = i
                        
                if begin_reactant != end_reactant and begin_reactant is not None and end_reactant is not None:
                    new_cn_bonds += 1
                    
            return new_cn_bonds >= 1  # At least one new intermolecular C-N bond formed
            
        except Exception:
            return False
