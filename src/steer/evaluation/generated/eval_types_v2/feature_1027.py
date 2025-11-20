"""Generated evaluation code for: Late stage ether formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageEtherFormation(BaseScoring):
    """
    Evaluates whether ether formation occurs at late stage (within depth threshold from target).
    Detects C-O-C ether bond formation reactions and rewards when they happen late in synthesis.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 2)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No ether formation found
        
        # Convert depth fraction to score where late stage (low depth) gets higher score
        # x is depth fraction (0 = at target, 1 = at root)
        if x <= self.depth_threshold / 10.0:  # Within threshold
            return 10 * (1 - x)  # Higher score for later reactions
        else:
            return max(0, 5 * (1 - x))  # Reduced score for earlier reactions
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves ether formation (C-O-C bond formation)"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles.strip())
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Count C-O-C ethers in product vs reactants
            product_ethers = self._count_ether_bonds(product)
            reactant_ethers = sum(self._count_ether_bonds(r) for r in reactants)
            
            # Ether formation if product has more C-O-C bonds than reactants
            return product_ethers > reactant_ethers
            
        except Exception:
            return False
    
    def _count_ether_bonds(self, mol) -> int:
        """Count C-O-C ether bonds in molecule"""
        if not mol:
            return 0
            
        count = 0
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()
                
                # Check for C-O bond where O is connected to another C
                if atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'O':
                    # Check if oxygen has another carbon neighbor
                    o_neighbors = [n for n in atom2.GetNeighbors() if n.GetSymbol() == 'C']
                    if len(o_neighbors) >= 2:  # O connected to at least 2 carbons
                        count += 1
                elif atom1.GetSymbol() == 'O' and atom2.GetSymbol() == 'C':
                    # Check if oxygen has another carbon neighbor
                    o_neighbors = [n for n in atom1.GetNeighbors() if n.GetSymbol() == 'C']
                    if len(o_neighbors) >= 2:  # O connected to at least 2 carbons
                        count += 1
        
        # Divide by 2 since we count each ether bond twice
        return count // 2
