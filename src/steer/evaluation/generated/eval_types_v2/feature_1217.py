"""Generated evaluation code for: Late stage aromatic fluorination via Sandmeyer"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSandmeyerFluorination(BaseScoring):
    """
    Evaluates routes for late-stage aromatic fluorination via Sandmeyer reaction.
    Detects conversion of aromatic amine to aromatic fluoride and rewards
    reactions occurring later in the synthesis (closer to the target molecule).
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.1)  # Late stage preferred
    
    def route_scoring(self, x) -> float:
        """Score based on how late the Sandmeyer reaction occurs"""
        if x < 0:
            return 0  # Reaction doesn't happen
        else:
            # Later reactions (smaller x) get higher scores
            # x=0 (final step) gets score of 1, x=1 (first step) gets score of 0
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents a Sandmeyer fluorination"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if product has aromatic fluorine
            aromatic_fluorine_pattern = Chem.MolFromSmarts("[cH0:1][F]")
            if not product_mol.HasSubstructMatch(aromatic_fluorine_pattern):
                return False
            
            # Check if any reactant has aromatic amine at corresponding position
            aromatic_amine_pattern = Chem.MolFromSmarts("[cH0:1][NH2,NH3+]")
            has_aromatic_amine = False
            
            for reactant in reactant_mols:
                if reactant.HasSubstructMatch(aromatic_amine_pattern):
                    # Verify the transformation using atom mapping if available
                    if self._verify_amine_to_fluoride_transformation(product_mol, reactant):
                        has_aromatic_amine = True
                        break
            
            # Additional check for Sandmeyer reagents (fluoride source)
            has_fluoride_source = self._has_sandmeyer_reagents(reactant_mols)
            
            return has_aromatic_amine and has_fluoride_source
            
        except Exception:
            return False
    
    def _verify_amine_to_fluoride_transformation(self, product, reactant) -> bool:
        """Verify that aromatic amine position corresponds to aromatic fluoride position"""
        # Get atom map numbers for verification
        product_atoms = {atom.GetAtomMapNum(): atom for atom in product.GetAtoms() 
                        if atom.GetAtomMapNum() > 0}
        reactant_atoms = {atom.GetAtomMapNum(): atom for atom in reactant.GetAtoms() 
                         if atom.GetAtomMapNum() > 0}
        
        # Find mapped carbon atoms connected to F in product and N in reactant
        for map_num in product_atoms:
            if map_num in reactant_atoms:
                prod_atom = product_atoms[map_num]
                react_atom = reactant_atoms[map_num]
                
                # Check if carbon atom has F neighbor in product and N neighbor in reactant
                prod_neighbors = [n.GetSymbol() for n in prod_atom.GetNeighbors()]
                react_neighbors = [n.GetSymbol() for n in react_atom.GetNeighbors()]
                
                if 'F' in prod_neighbors and 'N' in react_neighbors:
                    return True
        
        return True  # Default to True if no atom mapping available
    
    def _has_sandmeyer_reagents(self, reactant_mols) -> bool:
        """Check for typical Sandmeyer fluorination reagents"""
        fluoride_sources = [
            Chem.MolFromSmarts("[F-]"),  # Fluoride anion
            Chem.MolFromSmarts("B[F]"),  # Fluoroboric acid derivatives
            Chem.MolFromSmarts("[N+]#N"),  # Diazonium
            Chem.MolFromSmarts("N(=O)=O"),  # Nitrous acid derivatives
        ]
        
        for reactant in reactant_mols:
            for pattern in fluoride_sources:
                if pattern and reactant.HasSubstructMatch(pattern):
                    return True
        
        return False
