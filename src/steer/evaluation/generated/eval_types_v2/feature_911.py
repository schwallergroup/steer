"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two major fragments are prepared 
    separately and then coupled in a late-stage reaction. Detects C-H arylation 
    and other common coupling reactions that join two substantial fragments.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_position = config.get("coupling_step_position", "late")
        
        # Common coupling reaction patterns (C-H arylation, Suzuki, etc.)
        self.coupling_patterns = [
            # C-H arylation patterns
            "[cH:1]>>[c:1]-[c]",  # Aromatic C-H to C-C
            "[CH:1]>>[C:1]-[c]",  # Aliphatic C-H to C-aryl
            # Cross-coupling patterns
            "[c:1]-[Br,I,Cl].[c:2]-[B]>>[c:1]-[c:2]",  # Suzuki-type
            "[c:1]-[Br,I,Cl].[c:2]-[Sn]>>[c:1]-[c:2]", # Stille-type
        ]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No convergent coupling found
        
        if self.coupling_position == "late":
            # Reward later coupling (lower depth fraction is better)
            return max(0, 10 * (1 - x))
        else:
            # For other positions, could implement different scoring
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling of two fragments.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            reactants = react_smiles.split(".")
            
            # Must have exactly the target number of reactants (fragments)
            if len(reactants) != self.fragment_count:
                return False
            
            # Check if this looks like a coupling reaction
            if self._is_coupling_reaction(prod_smiles, reactants):
                # Verify fragments are substantial (not just small reagents)
                if self._are_substantial_fragments(reactants):
                    return True
                    
        except Exception:
            pass
            
        return False
    
    def _is_coupling_reaction(self, product_smiles: str, reactant_smiles: list) -> bool:
        """Check if reaction matches coupling patterns."""
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles if Chem.MolFromSmiles(r)]
            
            if not product or len(reactants) != len(reactant_smiles):
                return False
            
            # Simple heuristic: product should be larger than individual reactants
            prod_atoms = product.GetNumAtoms()
            reactant_atoms = [mol.GetNumAtoms() for mol in reactants]
            
            # Product should be roughly sum of reactants (allowing for leaving groups)
            total_reactant_atoms = sum(reactant_atoms)
            if prod_atoms < total_reactant_atoms * 0.7:  # Allow for leaving groups
                return False
                
            # Check for new C-C bonds (simplified check)
            return self._has_new_cc_bond(product, reactants)
            
        except Exception:
            return False
    
    def _has_new_cc_bond(self, product, reactants) -> bool:
        """Check if new C-C bonds were formed."""
        try:
            # Count aromatic C-C bonds in product vs reactants
            prod_cc_bonds = self._count_aromatic_cc_bonds(product)
            reactant_cc_bonds = sum(self._count_aromatic_cc_bonds(r) for r in reactants)
            
            # New C-C bond formation
            return prod_cc_bonds > reactant_cc_bonds
            
        except Exception:
            return False
    
    def _count_aromatic_cc_bonds(self, mol) -> int:
        """Count C-C bonds involving aromatic carbons."""
        count = 0
        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            if (begin_atom.GetSymbol() == 'C' and end_atom.GetSymbol() == 'C' and 
                (begin_atom.GetIsAromatic() or end_atom.GetIsAromatic())):
                count += 1
        return count
    
    def _are_substantial_fragments(self, reactant_smiles: list) -> bool:
        """Check if reactants are substantial fragments, not just small reagents."""
        try:
            min_atoms = 6  # Minimum atoms to consider substantial
            substantial_count = 0
            
            for smiles in reactant_smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol and mol.GetNumAtoms() >= min_atoms:
                    # Check if it contains aromatic rings or substantial aliphatic chains
                    if any(atom.GetIsAromatic() for atom in mol.GetAtoms()) or mol.GetNumAtoms() >= 8:
                        substantial_count += 1
            
            return substantial_count >= self.fragment_count
            
        except Exception:
            return False
