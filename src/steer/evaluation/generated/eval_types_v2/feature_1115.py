"""Generated evaluation code for: Late stage Michael addition for final assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageMichaelAddition(BaseScoring):
    """
    Evaluates if a Michael addition reaction occurs in the final stages of synthesis.
    Checks for conjugate addition patterns within the specified depth range (0-2).
    """
    
    def __init__(self, config: Dict):
        self.depth_range = config["parameters"]["depth_range"]
        self.max_allowed_depth = self.depth_range[1] / 10.0  # Convert to fraction
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Michael addition doesn't occur
        
        # Convert depth fraction back to actual depth
        actual_depth = x * 10
        
        # Check if within desired range
        if actual_depth <= self.depth_range[1]:
            # Earlier is better within the allowed range
            return 10 - (actual_depth * 2)  # Scale to 0-10
        else:
            # Penalize if too late
            return max(0, 10 - actual_depth)
    
    def hit_condition(self, d) -> bool:
        """
        Detects Michael addition by looking for conjugate addition patterns.
        Identifies formation of C-C bonds adjacent to electron-withdrawing groups.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        try:
            prod_mol = Chem.MolFromSmiles(products)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            
            if not prod_mol or not all(reactant_mols):
                return False
            
            # Michael addition patterns to detect
            # Alpha,beta-unsaturated carbonyl acceptors
            michael_acceptor_patterns = [
                "[#6]=[#6]-[#6](=[#8])",  # α,β-unsaturated ketone
                "[#6]=[#6]-[#6](=[#8])-[#8]",  # α,β-unsaturated ester
                "[#6]=[#6]-[#6](=[#8])-[#7]",  # α,β-unsaturated amide
                "[#6]=[#6]-[#6]#[#7]",  # α,β-unsaturated nitrile
                "[#6]=[#6]-[#16](=[#8])(=[#8])",  # α,β-unsaturated sulfone
            ]
            
            # Check if reactants contain Michael acceptor and nucleophile
            has_acceptor = False
            has_nucleophile = False
            
            for reactant in reactant_mols:
                # Check for Michael acceptor patterns
                for pattern in michael_acceptor_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_acceptor = True
                        break
                
                # Check for nucleophile patterns (enolates, malonates, etc.)
                nucleophile_patterns = [
                    "[#6]-[#6](=[#8])-[#6]-[#6](=[#8])",  # malonate
                    "[#6]-[#6](=[#8])-[#6]",  # enolate precursor
                    "[#7]-[#6](=[#8])-[#6]",  # amide enolate
                ]
                
                for nuc_pattern in nucleophile_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(nuc_pattern)):
                        has_nucleophile = True
                        break
            
            # Additional check: look for newly formed C-C bond in product
            # that wasn't present in reactants
            if has_acceptor and has_nucleophile:
                return self._verify_bond_formation(prod_mol, reactant_mols)
                
            return False
            
        except Exception:
            return False
    
    def _verify_bond_formation(self, product, reactants) -> bool:
        """
        Verify that a new C-C bond is formed consistent with Michael addition.
        """
        try:
            # Get atom map numbers for tracking
            prod_atoms = {atom.GetAtomMapNum(): atom.GetIdx() 
                         for atom in product.GetAtoms() 
                         if atom.GetAtomMapNum() > 0}
            
            reactant_atoms = {}
            for reactant in reactants:
                for atom in reactant.GetAtoms():
                    if atom.GetAtomMapNum() > 0:
                        reactant_atoms[atom.GetAtomMapNum()] = reactant
            
            # Check for new C-C bonds in product
            for bond in product.GetBonds():
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()
                
                if (atom1.GetAtomicNum() == 6 and atom2.GetAtomicNum() == 6 and
                    atom1.GetAtomMapNum() > 0 and atom2.GetAtomMapNum() > 0):
                    
                    map1, map2 = atom1.GetAtomMapNum(), atom2.GetAtomMapNum()
                    
                    # Check if these atoms were in different reactants
                    if (map1 in reactant_atoms and map2 in reactant_atoms and
                        reactant_atoms[map1] != reactant_atoms[map2]):
                        return True
            
            return False
            
        except Exception:
            return False
