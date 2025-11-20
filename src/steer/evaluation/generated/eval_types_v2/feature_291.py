"""Generated evaluation code for: Late stage amide bond formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideBondFormation(BaseScoring):
    """
    Evaluates if amide bond formation occurs at late stage (specific step position from end).
    Checks for amide coupling reactions at the specified step position from the target molecule.
    """
    
    def __init__(self, config: Dict):
        self.step_position = config["parameters"]["step_position"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Amide coupling doesn't happen
        else:
            # Perfect score if at exact target position, decreasing otherwise
            target_fraction = self.step_position / 10.0  # Convert step to fraction
            return max(0, 1 - abs(x - target_fraction))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents an amide coupling reaction."""
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        product = Chem.MolFromSmiles(rxn_parts[0])
        reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
        
        if not product or not all(reactants):
            return False
            
        return self._is_amide_coupling_reaction(product, reactants)
    
    def _is_amide_coupling_reaction(self, product, reactants):
        """Detect if this is an amide bond formation reaction."""
        # Amide pattern: C(=O)N
        amide_pattern = Chem.MolFromSmarts("[C](=[O])[N]")
        
        if not product.HasSubstructMatch(amide_pattern):
            return False
            
        # Check if amide bond is newly formed (not present in reactants)
        product_amide_atoms = self._get_amide_atoms(product)
        
        for reactant in reactants:
            reactant_amide_atoms = self._get_amide_atoms(reactant)
            # If any reactant already has the same amide bond, this isn't formation
            if self._has_matching_amide_bond(product_amide_atoms, reactant_amide_atoms, product, reactant):
                return False
                
        # Additional check: look for typical amide coupling reagents/conditions
        return self._has_coupling_reagents(reactants)
    
    def _get_amide_atoms(self, mol):
        """Get atom map numbers involved in amide bonds."""
        amide_pattern = Chem.MolFromSmarts("[C](=[O])[N]")
        amide_bonds = []
        
        matches = mol.GetSubstructMatches(amide_pattern)
        for match in matches:
            c_idx, o_idx, n_idx = match
            c_mapnum = mol.GetAtomWithIdx(c_idx).GetAtomMapNum()
            n_mapnum = mol.GetAtomWithIdx(n_idx).GetAtomMapNum()
            if c_mapnum > 0 and n_mapnum > 0:
                amide_bonds.append((c_mapnum, n_mapnum))
                
        return amide_bonds
    
    def _has_matching_amide_bond(self, prod_amides, react_amides, product, reactant):
        """Check if product amide bond already exists in reactant."""
        for prod_amide in prod_amides:
            if prod_amide in react_amides:
                return True
        return False
    
    def _has_coupling_reagents(self, reactants):
        """Check for presence of typical amide coupling reagents or activated species."""
        # Look for activated acids (acid chlorides, anhydrides) or coupling reagents
        acid_chloride = Chem.MolFromSmarts("[C](=[O])[Cl]")
        activated_ester = Chem.MolFromSmarts("[C](=[O])[O][C]")
        carboxylic_acid = Chem.MolFromSmarts("[C](=[O])[OH]")
        amine = Chem.MolFromSmarts("[N;H1,H2]")
        
        has_acid_component = False
        has_amine_component = False
        
        for reactant in reactants:
            if (reactant.HasSubstructMatch(acid_chloride) or 
                reactant.HasSubstructMatch(activated_ester) or 
                reactant.HasSubstructMatch(carboxylic_acid)):
                has_acid_component = True
            if reactant.HasSubstructMatch(amine):
                has_amine_component = True
                
        return has_acid_component and has_amine_component
