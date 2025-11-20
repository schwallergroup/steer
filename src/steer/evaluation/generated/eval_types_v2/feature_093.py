"""Generated evaluation code for: Early stage ether formation via Williamson synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class WilliamsonEtherSynthesis(BaseScoring):
    """
    Evaluates whether Williamson ether synthesis occurs in early stages of the route.
    
    Williamson ether synthesis involves nucleophilic substitution of an alkoxide
    with an alkyl halide or tosylate to form an ether bond (R-O-R').
    """
    
    def __init__(self, config: Dict):
        self.stage_cutoff = config["parameters"]["stage_cutoff"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't happen
        elif x <= self.stage_cutoff:
            return 10  # Early stage - excellent score
        else:
            # Penalize later stages linearly
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Detect Williamson ether synthesis by checking for:
        1. Formation of new C-O-C ether bond
        2. Presence of leaving group (halide, tosylate, etc.) in reactants
        3. Alkoxide or phenoxide nucleophile
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Check for new ether formation
            if not self._has_new_ether_bond(reactants, product):
                return False
                
            # Check for typical Williamson reactants
            has_alkoxide = any(self._is_alkoxide_precursor(r) for r in reactants)
            has_leaving_group = any(self._has_leaving_group(r) for r in reactants)
            
            return has_alkoxide and has_leaving_group
            
        except Exception:
            return False
    
    def _has_new_ether_bond(self, reactants, product) -> bool:
        """Check if new C-O-C ether bonds are formed"""
        # Count ether bonds in reactants vs product
        reactant_ethers = sum(self._count_ether_bonds(mol) for mol in reactants)
        product_ethers = self._count_ether_bonds(product)
        
        return product_ethers > reactant_ethers
    
    def _count_ether_bonds(self, mol) -> int:
        """Count C-O-C ether bonds (excluding C=O, acids, esters)"""
        if not mol:
            return 0
            
        count = 0
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
                
                # Check for C-O-C pattern
                if ((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'O') or 
                    (atom1.GetSymbol() == 'O' and atom2.GetSymbol() == 'C')):
                    
                    oxygen = atom1 if atom1.GetSymbol() == 'O' else atom2
                    carbon = atom2 if atom1.GetSymbol() == 'O' else atom1
                    
                    # Exclude carbonyls, carboxylic acids, esters
                    if (oxygen.GetTotalNumHs() == 0 and 
                        len([n for n in oxygen.GetNeighbors() if n.GetSymbol() == 'C']) == 2):
                        # Simple ether: oxygen connected to exactly 2 carbons
                        carbon_neighbors = [n for n in oxygen.GetNeighbors() if n.GetSymbol() == 'C']
                        if len(carbon_neighbors) == 2:
                            # Check neither carbon is carbonyl
                            is_ether = True
                            for c in carbon_neighbors:
                                for c_neighbor in c.GetNeighbors():
                                    if (c_neighbor.GetSymbol() == 'O' and 
                                        mol.GetBondBetweenAtoms(c.GetIdx(), c_neighbor.GetIdx()).GetBondType() == Chem.BondType.DOUBLE):
                                        is_ether = False
                                        break
                            if is_ether:
                                count += 0.5  # Each ether bond counted from both directions
        
        return int(count)
    
    def _is_alkoxide_precursor(self, mol) -> bool:
        """Check for alkoxide, phenoxide, or alcohol precursors"""
        if not mol:
            return False
            
        # Check for phenol/alcohol OH groups
        phenol_pattern = Chem.MolFromSmarts("[OH1][c,C]")
        if mol.HasSubstructMatch(phenol_pattern):
            return True
            
        # Check for alkoxide salts (simplified - look for O- patterns)
        # This is approximate since charge info may not be preserved
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() == 0:
                carbon_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() in ['C', 'c']]
                if len(carbon_neighbors) == 1:  # Terminal oxygen
                    return True
                    
        return False
    
    def _has_leaving_group(self, mol) -> bool:
        """Check for common leaving groups (halides, tosylates, mesylates)"""
        if not mol:
            return False
            
        # Halides
        halide_pattern = Chem.MolFromSmarts("[C,c][F,Cl,Br,I]")
        if mol.HasSubstructMatch(halide_pattern):
            return True
            
        # Tosylates
        tosylate_pattern = Chem.MolFromSmarts("CS(=O)(=O)O[C,c]")
        if mol.HasSubstructMatch(tosylate_pattern):
            return True
            
        # Mesylates
        mesylate_pattern = Chem.MolFromSmarts("[CH3]S(=O)(=O)O[C,c]")
        if mol.HasSubstructMatch(mesylate_pattern):
            return True
            
        return False
