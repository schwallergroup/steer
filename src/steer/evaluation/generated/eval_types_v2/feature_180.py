"""Generated evaluation code for: Convergent synthesis via two major fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by checking if the route assembles 
    major fragments separately and couples them in a final step.
    
    Looks for reactions where multiple substantial fragments (>= min_atoms) 
    are coupled together, indicating convergent assembly.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_step = config.get("coupling_step", "final")
        self.min_atoms = config.get("min_atoms_per_fragment", 6)
        # Common coupling reaction patterns
        self.coupling_patterns = [
            "[#6]-[#6]",  # C-C bond formation
            "[#6]-[#7]",  # C-N bond formation  
            "[#6]-[#8]",  # C-O bond formation
            "[#6]-[#16]", # C-S bond formation
        ]

    def route_scoring(self, x) -> float:
        """
        Score based on when convergent coupling occurs.
        Early convergent coupling (low depth) gets higher score.
        """
        if x < 0:
            return 0  # No convergent coupling found
        
        if self.coupling_step == "final":
            # Reward very early coupling (close to target)
            return max(0, 10 - (x * 8))
        else:
            # More flexible - any convergent step is good
            return max(0, 8 - (x * 6))

    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents convergent coupling of major fragments.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, product_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            product = Chem.MolFromSmiles(product_smiles.strip())
            
            if not product or len(reactants) < self.fragment_count:
                return False
                
            # Filter out small reactants (catalysts, reagents)
            major_reactants = []
            for r in reactants:
                if r and r.GetNumAtoms() >= self.min_atoms:
                    major_reactants.append(r)
            
            # Check if we have the required number of major fragments
            if len(major_reactants) < self.fragment_count:
                return False
                
            # Verify this looks like a coupling reaction
            return self._is_coupling_reaction(major_reactants, product)
            
        except Exception:
            return False

    def _is_coupling_reaction(self, reactants, product) -> bool:
        """
        Check if the reaction appears to be coupling major fragments together.
        """
        # Check that product is significantly larger than individual reactants
        product_atoms = product.GetNumAtoms()
        total_reactant_atoms = sum(r.GetNumAtoms() for r in reactants)
        
        # Product should contain most atoms from reactants (allowing for small leaving groups)
        if product_atoms < total_reactant_atoms * 0.8:
            return False
            
        # Check for formation of new bonds between fragments
        return self._has_new_interfragment_bonds(reactants, product)
    
    def _has_new_interfragment_bonds(self, reactants, product) -> bool:
        """
        Check if new bonds are formed between the major fragments.
        """
        # Get atom mappings from reactants
        reactant_maps = []
        for r in reactants:
            r_maps = set()
            for atom in r.GetAtoms():
                if atom.GetAtomMapNum() > 0:
                    r_maps.add(atom.GetAtomMapNum())
            reactant_maps.append(r_maps)
        
        # Look for bonds in product between atoms from different reactants
        for bond in product.GetBonds():
            atom1_map = bond.GetBeginAtom().GetAtomMapNum()
            atom2_map = bond.GetEndAtom().GetAtomMapNum()
            
            if atom1_map > 0 and atom2_map > 0:
                # Check if these atoms came from different reactants
                atom1_reactant = None
                atom2_reactant = None
                
                for i, r_maps in enumerate(reactant_maps):
                    if atom1_map in r_maps:
                        atom1_reactant = i
                    if atom2_map in r_maps:
                        atom2_reactant = i
                
                # If atoms from different major reactants are bonded, it's convergent
                if (atom1_reactant is not None and atom2_reactant is not None 
                    and atom1_reactant != atom2_reactant):
                    return True
                    
        return False
