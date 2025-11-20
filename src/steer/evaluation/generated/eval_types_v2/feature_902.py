"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis routes where two complex fragments are built 
    separately and coupled via specific reaction types (Williamson ether formation, 
    amide coupling, etc.).
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["parameters"]["fragment_count"]
        self.coupling_reactions = config["parameters"]["coupling_reactions"]
        
        # Define SMARTS patterns for coupling reactions
        self.reaction_patterns = {
            "williamson_ether": "[C,c]-O-[C,c]",  # Ether linkage
            "amide_coupling": "[C,c]-C(=O)-N-[C,c]",  # Amide bond
            "suzuki_coupling": "[c]-[c]",  # Aryl-aryl bond (simplified)
            "click_chemistry": "[c,C]-[nH0]=[nH0]-[nH0]-[c,C]",  # Triazole from click
            "reductive_amination": "[C,c]-N-[C,c]",  # C-N bond formation
        }

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No convergent coupling found
        else:
            # Earlier convergent coupling is better (more balanced synthesis)
            return 1 - x

    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling between 
        two complex fragments using specified coupling reactions.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactant_smiles_list = reactants_smiles.split(".")
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles_list if r]
            
            if not product or len(reactants) != self.fragment_count:
                return False
                
            # Check if reactants are sufficiently complex (at least 6 heavy atoms each)
            min_complexity = 6
            if not all(mol.GetNumHeavyAtoms() >= min_complexity for mol in reactants):
                return False
                
            # Check for specified coupling reaction patterns
            return self._check_coupling_patterns(product, reactants, mapped_rxn)
            
        except Exception:
            return False

    def _check_coupling_patterns(self, product, reactants, mapped_rxn):
        """Check if the reaction involves one of the specified coupling patterns."""
        
        for coupling_type in self.coupling_reactions:
            if coupling_type in self.reaction_patterns:
                pattern = self.reaction_patterns[coupling_type]
                
                # Check if the coupling pattern exists in product
                if self._pattern_exists_in_product(product, pattern):
                    # Verify this linkage was formed in this step
                    if self._linkage_formed_in_reaction(pattern, product, reactants, mapped_rxn):
                        return True
                        
        return False

    def _pattern_exists_in_product(self, product, pattern):
        """Check if the coupling pattern exists in the product."""
        try:
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol:
                return product.HasSubstructMatch(pattern_mol)
        except Exception:
            pass
        return False

    def _linkage_formed_in_reaction(self, pattern, product, reactants, mapped_rxn):
        """
        Verify that the coupling linkage was actually formed in this reaction
        by checking that the pattern exists in product but not in individual reactants.
        """
        try:
            pattern_mol = Chem.MolFromSmarts(pattern)
            if not pattern_mol:
                return False
                
            # Pattern should exist in product
            if not product.HasSubstructMatch(pattern_mol):
                return False
                
            # For convergent coupling, the complete linkage pattern should not 
            # exist in any single reactant (it's formed by joining them)
            for reactant in reactants:
                if reactant.HasSubstructMatch(pattern_mol):
                    # If pattern already exists in a reactant, this might not be 
                    # the convergent coupling step we're looking for
                    continue
                    
            # Additional check: verify atom mapping shows bond formation
            return self._verify_bond_formation_by_mapping(mapped_rxn, pattern)
            
        except Exception:
            return False
            
        return True

    def _verify_bond_formation_by_mapping(self, mapped_rxn, pattern):
        """
        Use atom mapping to verify that a new bond corresponding to the 
        coupling pattern was formed between fragments.
        """
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            # Get atom map numbers from product and reactants
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            # Get all atom map numbers in reactants
            reactant_maps = set()
            for reactant in reactants:
                for atom in reactant.GetAtoms():
                    if atom.GetAtomMapNum():
                        reactant_maps.add(atom.GetAtomMapNum())
            
            # Check bonds in product between atoms from different reactants
            for bond in product.GetBonds():
                begin_map = bond.GetBeginAtom().GetAtomMapNum()
                end_map = bond.GetEndAtom().GetAtomMapNum()
                
                if begin_map and end_map and begin_map in reactant_maps and end_map in reactant_maps:
                    # Check if these atoms were in different reactants
                    if self._atoms_from_different_reactants(begin_map, end_map, reactants):
                        return True
                        
        except Exception:
            pass
            
        return True  # Default to True if mapping verification fails

    def _atoms_from_different_reactants(self, map1, map2, reactants):
        """Check if two mapped atoms come from different reactants."""
        reactant_containing_map1 = None
        reactant_containing_map2 = None
        
        for i, reactant in enumerate(reactants):
            maps_in_reactant = {atom.GetAtomMapNum() for atom in reactant.GetAtoms() if atom.GetAtomMapNum()}
            
            if map1 in maps_in_reactant:
                reactant_containing_map1 = i
            if map2 in maps_in_reactant:
                reactant_containing_map2 = i
                
        return (reactant_containing_map1 is not None and 
                reactant_containing_map2 is not None and 
                reactant_containing_map1 != reactant_containing_map2)
