"""Generated evaluation code for: Late stage intramolecular cyclization via N-alkylation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageIntramolecularCyclization(BaseScoring):
    """
    Evaluates late-stage intramolecular cyclization via N-alkylation.
    Checks for formation of macrocyclic rings (7+ membered) through 
    intramolecular N-alkylation reactions occurring late in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.min_ring_size = config.get("min_ring_size", 7)  # Macrocycle threshold
        self.condition_type = config["target_depth"]["type"]
        self.target_depth = config["target_depth"]["value"]
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0  # No intramolecular cyclization found
            # Late-stage cyclization is better (closer to 1.0)
            return max(0, 1 - abs(x - self.target_depth))
    
    def hit_condition(self, d):
        """
        Checks if a reaction involves intramolecular N-alkylation 
        leading to macrocycle formation.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if new ring is formed (product has more rings than largest reactant)
            product_rings = self._get_ring_info(product)
            max_reactant_rings = max([self._get_ring_info(r) for r in reactants], default=[])
            
            # Look for new macrocyclic rings
            new_macrocycles = []
            for ring_size in product_rings:
                if ring_size >= self.min_ring_size:
                    if not max_reactant_rings or ring_size not in max_reactant_rings:
                        new_macrocycles.append(ring_size)
            
            if not new_macrocycles:
                return False
            
            # Check for intramolecular N-alkylation pattern
            return self._is_intramolecular_n_alkylation(reactants_smiles, products_smiles)
            
        except Exception:
            return False
    
    def _get_ring_info(self, mol):
        """Get list of ring sizes in molecule."""
        if not mol:
            return []
        
        ring_info = mol.GetRingInfo()
        return [len(ring) for ring in ring_info.AtomRings()]
    
    def _is_intramolecular_n_alkylation(self, reactants_smiles, products_smiles):
        """
        Check if reaction represents intramolecular N-alkylation.
        Look for patterns indicating nitrogen attacking an electrophilic carbon
        within the same molecule to form a ring.
        """
        try:
            # Parse with atom mapping to track atom changes
            product = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            # Look for nitrogen atoms that gain bonds in the product
            product_atom_maps = {atom.GetAtomMapNum(): atom for atom in product.GetAtoms() 
                               if atom.GetAtomMapNum() > 0}
            
            reactant_atom_maps = {}
            for reactant in reactants:
                for atom in reactant.GetAtoms():
                    if atom.GetAtomMapNum() > 0:
                        reactant_atom_maps[atom.GetAtomMapNum()] = atom
            
            # Check for nitrogen atoms that form new C-N bonds
            for map_num, prod_atom in product_atom_maps.items():
                if prod_atom.GetSymbol() == 'N' and map_num in reactant_atom_maps:
                    react_atom = reactant_atom_maps[map_num]
                    
                    # Count C-N bonds in reactant vs product
                    react_cn_bonds = sum(1 for neighbor in react_atom.GetNeighbors() 
                                       if neighbor.GetSymbol() == 'C')
                    prod_cn_bonds = sum(1 for neighbor in prod_atom.GetNeighbors() 
                                      if neighbor.GetSymbol() == 'C')
                    
                    # New C-N bond formed
                    if prod_cn_bonds > react_cn_bonds:
                        # Check if this is intramolecular (both atoms in same reactant)
                        if len(reactants) == 1:  # Single reactant = intramolecular
                            return True
                        
                        # For multiple reactants, check if N and new C partner 
                        # were in the same starting material
                        new_carbon_maps = self._find_new_carbon_partners(
                            map_num, reactants, product)
                        if self._atoms_in_same_reactant(map_num, new_carbon_maps, reactants):
                            return True
            
            return False
            
        except Exception:
            return False
    
    def _find_new_carbon_partners(self, n_map_num, reactants, product):
        """Find carbon atoms that form new bonds with the nitrogen."""
        new_partners = []
        
        # Get nitrogen's neighbors in product
        for atom in product.GetAtoms():
            if atom.GetAtomMapNum() == n_map_num:
                prod_carbon_neighbors = [n.GetAtomMapNum() for n in atom.GetNeighbors() 
                                       if n.GetSymbol() == 'C' and n.GetAtomMapNum() > 0]
                break
        else:
            return new_partners
        
        # Get nitrogen's neighbors in reactants
        react_carbon_neighbors = []
        for reactant in reactants:
            for atom in reactant.GetAtoms():
                if atom.GetAtomMapNum() == n_map_num:
                    react_carbon_neighbors.extend([n.GetAtomMapNum() for n in atom.GetNeighbors() 
                                                 if n.GetSymbol() == 'C' and n.GetAtomMapNum() > 0])
        
        # New carbon partners
        return [c for c in prod_carbon_neighbors if c not in react_carbon_neighbors]
    
    def _atoms_in_same_reactant(self, n_map, carbon_maps, reactants):
        """Check if nitrogen and carbon atoms were in the same reactant molecule."""
        for reactant in reactants:
            reactant_maps = [atom.GetAtomMapNum() for atom in reactant.GetAtoms() 
                           if atom.GetAtomMapNum() > 0]
            if n_map in reactant_maps:
                return any(c_map in reactant_maps for c_map in carbon_maps)
        return False
