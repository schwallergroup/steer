"""Generated evaluation code for: Early stage nitrile alkylation for chain installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class NitrileAlkylationDepth(BaseScoring):
    """
    Evaluates synthesis routes for early-stage nitrile alkylation reactions.
    Detects alpha-alkylation of nitriles where a C-C bond is formed adjacent to the nitrile group,
    particularly for chain installation reactions.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)  # Early stage default
        self.min_chain_length = config.get("min_chain_length", 3)  # Minimum alkyl chain length
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        
        if self.condition_type == "bool":
            return 1 if x >= 0 else 0
        else:
            # Early stage is better, penalize late-stage alkylation
            if x <= self.target_depth:
                return 1.0
            else:
                # Linear penalty for later stages
                return max(0, 1.0 - (x - self.target_depth) / (1.0 - self.target_depth))
    
    def hit_condition(self, d):
        """
        Detects nitrile alpha-alkylation reactions by checking:
        1. Presence of nitrile group in product
        2. Formation of new C-C bond adjacent to nitrile
        3. Addition of alkyl chain of sufficient length
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
            reactants_smiles = rxn_parts[1].split(".")
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles if Chem.MolFromSmiles(r)]
            
            if not product_mol or not reactant_mols:
                return False
            
            # Check for nitrile in product
            nitrile_pattern = Chem.MolFromSmarts("[C]#N")
            if not product_mol.HasSubstructMatch(nitrile_pattern):
                return False
            
            # Find nitrile carbons in product with atom maps
            nitrile_matches = product_mol.GetSubstructMatches(nitrile_pattern)
            nitrile_carbon_maps = []
            
            for match in nitrile_matches:
                nitrile_c_atom = product_mol.GetAtomWithIdx(match[0])
                if nitrile_c_atom.GetAtomMapNum() > 0:
                    nitrile_carbon_maps.append(nitrile_c_atom.GetAtomMapNum())
            
            if not nitrile_carbon_maps:
                return False
            
            # Check if alpha-alkylation occurred
            for nitrile_map in nitrile_carbon_maps:
                if self._detect_alpha_alkylation(product_mol, reactant_mols, nitrile_map):
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _detect_alpha_alkylation(self, product_mol, reactant_mols, nitrile_carbon_map):
        """
        Detect if alpha-alkylation occurred at the nitrile carbon.
        """
        # Find the nitrile carbon in product
        nitrile_carbon = None
        for atom in product_mol.GetAtoms():
            if atom.GetAtomMapNum() == nitrile_carbon_map:
                nitrile_carbon = atom
                break
        
        if not nitrile_carbon:
            return False
        
        # Get alpha carbons (neighbors of nitrile carbon)
        alpha_carbons = []
        for neighbor in nitrile_carbon.GetNeighbors():
            if neighbor.GetSymbol() == "C" and neighbor.GetAtomMapNum() > 0:
                alpha_carbons.append(neighbor.GetAtomMapNum())
        
        # Check if any alpha carbon has a new alkyl chain
        for alpha_map in alpha_carbons:
            if self._has_new_alkyl_chain(product_mol, reactant_mols, alpha_map):
                return True
        
        return False
    
    def _has_new_alkyl_chain(self, product_mol, reactant_mols, alpha_carbon_map):
        """
        Check if the alpha carbon has a new alkyl chain that wasn't present in reactants.
        """
        # Find alpha carbon in product
        alpha_carbon_prod = None
        for atom in product_mol.GetAtoms():
            if atom.GetAtomMapNum() == alpha_carbon_map:
                alpha_carbon_prod = atom
                break
        
        if not alpha_carbon_prod:
            return False
        
        # Count alkyl substituents on alpha carbon in product
        prod_alkyl_chains = self._count_alkyl_chains(alpha_carbon_prod)
        
        # Count alkyl substituents on alpha carbon in reactants
        reactant_alkyl_chains = 0
        for reactant_mol in reactant_mols:
            for atom in reactant_mol.GetAtoms():
                if atom.GetAtomMapNum() == alpha_carbon_map:
                    reactant_alkyl_chains = self._count_alkyl_chains(atom)
                    break
        
        # New alkyl chain was added if product has more chains
        return prod_alkyl_chains > reactant_alkyl_chains
    
    def _count_alkyl_chains(self, carbon_atom):
        """
        Count the number of alkyl chains of sufficient length attached to a carbon.
        """
        alkyl_chains = 0
        
        for neighbor in carbon_atom.GetNeighbors():
            if neighbor.GetSymbol() == "C":
                chain_length = self._get_alkyl_chain_length(carbon_atom, neighbor, visited=set())
                if chain_length >= self.min_chain_length:
                    alkyl_chains += 1
        
        return alkyl_chains
    
    def _get_alkyl_chain_length(self, start_atom, current_atom, visited):
        """
        Recursively calculate the length of an alkyl chain.
        """
        if current_atom.GetIdx() in visited:
            return 0
        
        visited.add(current_atom.GetIdx())
        
        # Only count carbons in alkyl chains (not aromatic, not in rings > 6)
        if current_atom.GetSymbol() != "C" or current_atom.GetIsAromatic():
            return 0
        
        max_length = 1  # Current carbon
        
        for neighbor in current_atom.GetNeighbors():
            if neighbor.GetIdx() != start_atom.GetIdx() and neighbor.GetSymbol() == "C":
                length = 1 + self._get_alkyl_chain_length(current_atom, neighbor, visited.copy())
                max_length = max(max_length, length)
        
        return max_length
