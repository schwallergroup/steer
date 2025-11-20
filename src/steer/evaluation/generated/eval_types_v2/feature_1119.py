"""Generated evaluation code for: Benzylic cyanation for side chain installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylicCyanation(BaseScoring):
    """
    Evaluates routes for benzylic cyanation reactions involving C-Br bond breaking
    and C-C bond formation through nucleophilic substitution at benzylic positions.
    """
    
    def __init__(self, config: Dict):
        self.target_depth = config.get("depth", 8)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Earlier occurrence is better, normalize to 0-1 scale
            return max(0, 1 - x)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves benzylic cyanation via C-Br breaking"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants, products = rxn_smiles.split(">>")
            
            prod_mol = Chem.MolFromSmiles(reactants)  # Product in retro context
            react_mols = [Chem.MolFromSmiles(r) for r in products.split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
            
            # Check for C-Br bond breaking and C-C≡N formation
            return self._detect_benzylic_cyanation(prod_mol, react_mols)
            
        except:
            return False
    
    def _detect_benzylic_cyanation(self, product, reactants) -> bool:
        """Detect benzylic C-Br -> C-CN transformation"""
        
        # Pattern for benzylic carbon with cyano group
        benzylic_cn_pattern = Chem.MolFromSmarts("[cH0,cH1,cH2]C([CH2,CH3,CH0])C#N")
        
        # Pattern for benzylic bromide
        benzylic_br_pattern = Chem.MolFromSmarts("[cH0,cH1,cH2]C([CH2,CH3,CH0])Br")
        
        # Check if product has benzylic cyano group
        has_benzylic_cn = product.HasSubstructMatch(benzylic_cn_pattern)
        if not has_benzylic_cn:
            return False
        
        # Check if any reactant has corresponding benzylic bromide
        for reactant in reactants:
            if reactant.HasSubstructMatch(benzylic_br_pattern):
                # Verify atom mapping consistency
                if self._verify_atom_mapping(product, reactant):
                    return True
        
        return False
    
    def _verify_atom_mapping(self, product, reactant) -> bool:
        """Verify that the benzylic carbon maps correctly between product and reactant"""
        try:
            # Find benzylic carbons in both molecules using atom map numbers
            prod_benzylic = self._find_benzylic_carbon(product, has_cn=True)
            react_benzylic = self._find_benzylic_carbon(reactant, has_br=True)
            
            if prod_benzylic is None or react_benzylic is None:
                return False
            
            # Check if they have the same atom map number
            prod_map = prod_benzylic.GetAtomMapNum()
            react_map = react_benzylic.GetAtomMapNum()
            
            return prod_map > 0 and prod_map == react_map
            
        except:
            return False
    
    def _find_benzylic_carbon(self, mol, has_cn=False, has_br=False):
        """Find the benzylic carbon atom based on connectivity"""
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != 'C':
                continue
                
            # Check if carbon is connected to aromatic carbon
            aromatic_neighbor = False
            target_neighbor = False
            
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIsAromatic() and neighbor.GetSymbol() == 'C':
                    aromatic_neighbor = True
                elif has_cn and neighbor.GetSymbol() == 'C':
                    # Check if this carbon is part of cyano group
                    for nn in neighbor.GetNeighbors():
                        if nn.GetSymbol() == 'N' and neighbor.GetTotalDegree() == 2:
                            target_neighbor = True
                            break
                elif has_br and neighbor.GetSymbol() == 'Br':
                    target_neighbor = True
            
            if aromatic_neighbor and target_neighbor:
                return atom
                
        return None
