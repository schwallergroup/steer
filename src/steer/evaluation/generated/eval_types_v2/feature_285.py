"""Generated evaluation code for: Late stage reductive amination coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ReductiveAminationCoupling(BaseScoring):
    """
    Evaluates synthesis routes for late-stage reductive amination coupling reactions.
    Checks if a reductive amination occurs in the final steps to form C-N bonds,
    typically coupling an amine with an aldehyde or ketone.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "final_step")
        self.require_coupling = config.get("coupling", True)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reductive amination doesn't occur
        
        if self.timing == "final_step":
            # Reward later occurrence (closer to final step)
            return 10 * (1 - x)
        else:
            # For other timing preferences, could add different scoring logic
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Detects reductive amination reactions by looking for:
        1. Formation of C-N bond from C=O + N-H
        2. Presence of aldehyde/ketone reactant and amine reactant
        3. C-N bond in product that wasn't in reactants
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(products_smiles)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactant_mols):
                return False
            
            # Check for reductive amination pattern
            return self._is_reductive_amination(reactant_mols, product)
            
        except Exception:
            return False
    
    def _is_reductive_amination(self, reactants, product) -> bool:
        """
        Check if reaction represents reductive amination:
        - One reactant has C=O (aldehyde/ketone)
        - One reactant has N-H (primary/secondary amine)
        - Product has new C-N bond where C=O was reduced
        """
        # SMARTS patterns
        carbonyl_pattern = Chem.MolFromSmarts("[C:1]=[O:2]")
        amine_pattern = Chem.MolFromSmarts("[N:1][H:2]")
        secondary_amine_product = Chem.MolFromSmarts("[C:1][N:2]")
        
        # Check reactants for carbonyl and amine
        has_carbonyl = False
        has_amine = False
        carbonyl_carbon_map = None
        amine_nitrogen_map = None
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(carbonyl_pattern):
                has_carbonyl = True
                # Get atom map number of carbonyl carbon
                match = reactant.GetSubstructMatch(carbonyl_pattern)
                if match:
                    carbonyl_atom = reactant.GetAtomWithIdx(match[0])
                    carbonyl_carbon_map = carbonyl_atom.GetAtomMapNum()
            
            if reactant.HasSubstructMatch(amine_pattern):
                has_amine = True
                # Get atom map number of amine nitrogen
                match = reactant.GetSubstructMatch(amine_pattern)
                if match:
                    amine_atom = reactant.GetAtomWithIdx(match[0])
                    amine_nitrogen_map = amine_atom.GetAtomMapNum()
        
        if not (has_carbonyl and has_amine):
            return False
        
        # Check if product has C-N bond between the mapped atoms
        if carbonyl_carbon_map and amine_nitrogen_map:
            return self._atoms_are_bonded_in_product(product, carbonyl_carbon_map, amine_nitrogen_map)
        
        return False
    
    def _atoms_are_bonded_in_product(self, product, carbon_map_num, nitrogen_map_num) -> bool:
        """Check if atoms with given map numbers are bonded in the product"""
        carbon_atom = None
        nitrogen_atom = None
        
        for atom in product.GetAtoms():
            if atom.GetAtomMapNum() == carbon_map_num:
                carbon_atom = atom
            elif atom.GetAtomMapNum() == nitrogen_map_num:
                nitrogen_atom = atom
        
        if carbon_atom and nitrogen_atom:
            # Check if there's a bond between them
            bond = product.GetBondBetweenAtoms(carbon_atom.GetIdx(), nitrogen_atom.GetIdx())
            return bond is not None
        
        return False
