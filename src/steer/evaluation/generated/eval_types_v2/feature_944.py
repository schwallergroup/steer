"""Generated evaluation code for: Convergent synthesis via thiophenol-electrophile coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentThiophenolCoupling(BaseScoring):
    """
    Evaluates convergent synthesis routes that use thiophenol-electrophile coupling
    to form C-S bonds, combining two independently prepared fragments.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["parameters"].get("fragment_count", 2)
        self.coupling_bond_type = config["parameters"].get("coupling_bond_type", "C-S")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Coupling reaction doesn't happen
        else:
            # Earlier convergent coupling is better (more fragments prepared independently)
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents thiophenol-electrophile coupling."""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            prod = Chem.MolFromSmiles(rxn[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            if len(reactants) < 2:
                return False
                
            # Check for thiophenol substructure in reactants
            thiophenol_pattern = Chem.MolFromSmarts("c1ccccc1S")
            has_thiophenol = any(mol.HasSubstructMatch(thiophenol_pattern) for mol in reactants)
            
            if not has_thiophenol:
                return False
            
            # Check for C-S bond formation by comparing reactants to product
            # Look for new C-S bonds in product that weren't in reactants
            product_cs_bonds = self._get_cs_bonds(prod)
            reactant_cs_bonds = set()
            for reactant in reactants:
                reactant_cs_bonds.update(self._get_cs_bonds(reactant))
            
            # Check if new C-S bonds formed
            new_cs_bonds = product_cs_bonds - reactant_cs_bonds
            
            # Verify we have the right fragment count (2 main reactants for convergent synthesis)
            if len(reactants) >= self.fragment_count and len(new_cs_bonds) > 0:
                # Additional check: ensure one fragment contains thiophenol and another is electrophilic
                return self._verify_convergent_coupling(reactants, thiophenol_pattern)
                
            return False
            
        except Exception:
            return False
    
    def _get_cs_bonds(self, mol) -> set:
        """Extract C-S bonds from molecule using atom map numbers."""
        cs_bonds = set()
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            
            # Check for C-S bond
            atoms_symbols = sorted([atom1.GetSymbol(), atom2.GetSymbol()])
            if atoms_symbols == ['C', 'S']:
                map1 = atom1.GetAtomMapNum()
                map2 = atom2.GetAtomMapNum()
                if map1 > 0 and map2 > 0:
                    cs_bonds.add(tuple(sorted([map1, map2])))
        
        return cs_bonds
    
    def _verify_convergent_coupling(self, reactants, thiophenol_pattern) -> bool:
        """Verify this represents convergent synthesis with thiophenol and electrophile."""
        thiophenol_reactant = None
        electrophile_reactant = None
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(thiophenol_pattern):
                thiophenol_reactant = reactant
            else:
                # Check if this could be an electrophile (has leaving groups or unsaturation)
                electrophile_patterns = [
                    Chem.MolFromSmarts("[C,c][Cl,Br,I]"),  # Alkyl/aryl halides
                    Chem.MolFromSmarts("[C,c]OS(=O)(=O)"),  # Tosylates, mesylates
                    Chem.MolFromSmarts("[C,c]=O"),  # Carbonyls (for reductive coupling)
                ]
                
                if any(reactant.HasSubstructMatch(pattern) for pattern in electrophile_patterns):
                    electrophile_reactant = reactant
        
        # Return True if we found both a thiophenol and an electrophile
        return thiophenol_reactant is not None and electrophile_reactant is not None
