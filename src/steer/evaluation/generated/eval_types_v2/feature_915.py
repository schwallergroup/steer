"""Generated evaluation code for: Methylsulfinyl leaving group activation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MethylsulfinylActivation(BaseScoring):
    """
    Detects sulfide oxidation reactions that create methylsulfinyl leaving groups
    for subsequent nucleophilic aromatic substitution reactions.
    
    Looks for the conversion of methylsulfide (C-S-CH3) to methylsulfoxide (C-S(=O)-CH3)
    attached to an aromatic system, which creates an excellent leaving group for SNAr.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Earlier oxidation is better for leaving group activation
            return 1 - x
    
    def hit_condition(self, d):
        """
        Check if this reaction represents sulfide to sulfoxide oxidation
        creating a methylsulfinyl leaving group on an aromatic ring.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in rxn[0].split(".")]
            products = [Chem.MolFromSmiles(p) for p in rxn[1].split(".")]
            
            # Pattern for aromatic methylsulfide (reactant)
            sulfide_pattern = Chem.MolFromSmarts("[c]-S-C")
            
            # Pattern for aromatic methylsulfoxide (product)
            sulfoxide_pattern = Chem.MolFromSmarts("[c]-S(=O)-C")
            
            # Check if we have sulfide in reactants and sulfoxide in products
            has_sulfide_reactant = any(mol.HasSubstructMatch(sulfide_pattern) for mol in reactants if mol is not None)
            has_sulfoxide_product = any(mol.HasSubstructMatch(sulfoxide_pattern) for mol in products if mol is not None)
            
            if has_sulfide_reactant and has_sulfoxide_product:
                # Verify this is actually an oxidation by checking atom mapping
                return self._verify_oxidation_mapping(reactants, products)
            
            return False
            
        except Exception:
            return False
    
    def _verify_oxidation_mapping(self, reactants, products):
        """
        Verify that the same sulfur atom is being oxidized by checking atom mapping.
        """
        try:
            # Find mapped sulfur atoms in reactants (should be sp3 hybridized)
            reactant_sulfurs = []
            for mol in reactants:
                if mol is not None:
                    for atom in mol.GetAtoms():
                        if (atom.GetSymbol() == 'S' and 
                            atom.GetAtomMapNum() > 0 and
                            len([n for n in atom.GetNeighbors()]) == 2):  # Two bonds for sulfide
                            reactant_sulfurs.append(atom.GetAtomMapNum())
            
            # Find mapped sulfur atoms in products (should have S=O bond)
            product_sulfurs = []
            for mol in products:
                if mol is not None:
                    for atom in mol.GetAtoms():
                        if (atom.GetSymbol() == 'S' and 
                            atom.GetAtomMapNum() > 0):
                            # Check if sulfur has double bond to oxygen
                            has_double_bond_o = any(
                                bond.GetBondType() == Chem.BondType.DOUBLE and 
                                bond.GetOtherAtom(atom).GetSymbol() == 'O'
                                for bond in atom.GetBonds()
                            )
                            if has_double_bond_o:
                                product_sulfurs.append(atom.GetAtomMapNum())
            
            # Check if same sulfur atom appears in both lists
            return bool(set(reactant_sulfurs) & set(product_sulfurs))
            
        except Exception:
            return False
