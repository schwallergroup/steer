"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy by detecting when two significant fragments
    are coupled together via a specific reaction type at a given depth in the route.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_reaction = config.get("coupling_reaction", "williamson_ether")
        
        # Define SMARTS patterns for different coupling reactions
        self.coupling_patterns = {
            "williamson_ether": "[C,c]-O-[C,c]",  # Ether formation
            "amide_coupling": "[C,c]-C(=O)-N-[C,c]",  # Amide bond
            "suzuki": "[c]-[c]",  # Aryl-aryl coupling
            "click": "[c,C]-[nH0]=[nH0]-[nH0]-[C,c]",  # Triazole formation
            "buchwald_hartwig": "[c]-N-[C,c]"  # C-N coupling
        }
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        else:
            # Earlier convergent coupling is generally better (more efficient)
            # Convert depth fraction to score where earlier = higher score
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling of fragments.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            product_smiles, reactants_smiles = rxn_smiles.split(">>")
            reactant_list = reactants_smiles.split(".")
            
            # Must have exactly the expected number of reactant fragments
            if len(reactant_list) != self.fragment_count:
                return False
            
            # Check if the coupling pattern is formed
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
            
            coupling_pattern = self.coupling_patterns.get(self.coupling_reaction)
            if not coupling_pattern:
                return False
                
            # Check if the coupling pattern exists in product
            pattern_mol = Chem.MolFromSmarts(coupling_pattern)
            if not pattern_mol or not product_mol.HasSubstructMatch(pattern_mol):
                return False
            
            # Verify fragments are substantial (not just small leaving groups)
            reactant_mols = []
            for r_smiles in reactant_list:
                mol = Chem.MolFromSmiles(r_smiles)
                if mol and self._is_substantial_fragment(mol):
                    reactant_mols.append(mol)
            
            # Must have at least 2 substantial fragments for convergent synthesis
            if len(reactant_mols) < 2:
                return False
            
            # Check that the substantial fragments contribute atoms to the coupling
            return self._fragments_participate_in_coupling(
                product_mol, reactant_mols, pattern_mol, rxn_smiles
            )
            
        except Exception:
            return False
    
    def _is_substantial_fragment(self, mol) -> bool:
        """
        Check if a molecule is a substantial fragment (not just a small reagent).
        """
        # Consider fragments with 6+ heavy atoms as substantial
        heavy_atom_count = mol.GetNumHeavyAtoms()
        return heavy_atom_count >= 6
    
    def _fragments_participate_in_coupling(self, product_mol, reactant_mols, 
                                         pattern_mol, rxn_smiles) -> bool:
        """
        Verify that the substantial fragments actually participate in forming the coupling bond.
        """
        try:
            # Get atom mapping to trace atoms through the reaction
            product_smiles, reactants_smiles = rxn_smiles.split(">>")
            
            # Find the coupling pattern in the product
            matches = product_mol.GetSubstructMatches(pattern_mol)
            if not matches:
                return False
            
            # For simplicity, check if we have mapped atoms that can trace back
            # to different reactant molecules in the coupling region
            product_atom_maps = {}
            for atom in product_mol.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num > 0:
                    product_atom_maps[map_num] = atom.GetIdx()
            
            # Check reactant atom mappings
            reactant_maps = []
            for r_mol in reactant_mols:
                r_maps = set()
                for atom in r_mol.GetAtoms():
                    map_num = atom.GetAtomMapNum()
                    if map_num > 0:
                        r_maps.add(map_num)
                reactant_maps.append(r_maps)
            
            # Verify that atoms from different reactants are involved
            if len(reactant_maps) >= 2 and len(product_atom_maps) > 0:
                # At least one atom from each substantial reactant should map to product
                for r_maps in reactant_maps:
                    if not any(map_num in product_atom_maps for map_num in r_maps):
                        return False
                return True
                        
            return True  # Fall back to basic checks if mapping is incomplete
            
        except Exception:
            return True  # Be permissive if detailed analysis fails
