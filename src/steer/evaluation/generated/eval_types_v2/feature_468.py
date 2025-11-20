"""Generated evaluation code for: Late piperidine ring formation via intramolecular cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePiperidineFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage piperidine ring formation via intramolecular cyclization.
    Checks if a piperidine ring (C1CCNCC1) is formed through intramolecular SN2 reaction at a late stage.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.formation_method = config["parameters"]["formation_method"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # For late-stage formation, lower depth fraction is better
            # Convert to 0-10 scale where late formation (low x) gets high score
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves piperidine ring formation via intramolecular cyclization
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactants):
                return False
            
            # Check if product contains piperidine ring
            if not product_mol.HasSubstructMatch(self.ring_pattern):
                return False
            
            # Check if this is intramolecular cyclization (single reactant forms ring)
            if len(reactants) != 1:
                return False
                
            reactant_mol = reactants[0]
            
            # Reactant should not have the piperidine ring (ring formation)
            if reactant_mol.HasSubstructMatch(self.ring_pattern):
                return False
            
            # Check for intramolecular SN2 pattern:
            # Look for nitrogen and carbon atoms that get connected
            # This is a simplified check for intramolecular cyclization
            return self._is_intramolecular_sn2_cyclization(reactant_mol, product_mol, mapped_rxn)
            
        except Exception:
            return False
    
    def _is_intramolecular_sn2_cyclization(self, reactant_mol, product_mol, mapped_rxn):
        """
        Check if the reaction represents an intramolecular SN2 cyclization to form piperidine
        """
        try:
            # Parse atom mappings to track bond formation
            rxn_parts = mapped_rxn.split(">>")
            
            # Get atom map numbers for nitrogen in piperidine ring in product
            piperidine_matches = product_mol.GetSubstructMatches(self.ring_pattern)
            if not piperidine_matches:
                return False
            
            # Get the first match of piperidine ring
            ring_atoms = piperidine_matches[0]
            
            # Find nitrogen atom in the ring (should be at index 3 based on SMARTS C1CCNCC1)
            nitrogen_idx = ring_atoms[3]
            nitrogen_atom = product_mol.GetAtomWithIdx(nitrogen_idx)
            nitrogen_mapnum = nitrogen_atom.GetAtomMapNum()
            
            if nitrogen_mapnum == 0:
                return False
            
            # Check if nitrogen in reactant has fewer bonds than in product
            # (indicating cyclization occurred)
            reactant_atoms = {atom.GetAtomMapNum(): atom for atom in reactant_mol.GetAtoms() if atom.GetAtomMapNum() > 0}
            
            if nitrogen_mapnum not in reactant_atoms:
                return False
            
            reactant_nitrogen = reactant_atoms[nitrogen_mapnum]
            
            # Simple heuristic: nitrogen should gain a bond during cyclization
            reactant_n_bonds = reactant_nitrogen.GetDegree()
            product_n_bonds = nitrogen_atom.GetDegree()
            
            return product_n_bonds > reactant_n_bonds
            
        except Exception:
            return False
