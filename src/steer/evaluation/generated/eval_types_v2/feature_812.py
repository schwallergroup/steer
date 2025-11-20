"""Generated evaluation code for: Late stage Williamson ether formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class WilliamsonEtherFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage Williamson ether formation.
    Detects C-O bond formation between a phenol and an alkyl halide/tosylate.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.1)  # Late stage preferred
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Williamson ether formation doesn't occur
        else:
            # Late-stage (low depth fraction) is better
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents Williamson ether formation"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for phenol pattern in reactants (ArOH)
            phenol_pattern = Chem.MolFromSmarts("[cH1,c]1[cH1,c][cH1,c][cH1,c]([OH1])[cH1,c][cH1,c]1")
            has_phenol = any(mol.HasSubstructMatch(phenol_pattern) for mol in reactants)
            
            # Check for alkyl halide or tosylate pattern
            alkyl_halide_pattern = Chem.MolFromSmarts("[CH2,CH3][Br,Cl,I]")
            tosylate_pattern = Chem.MolFromSmarts("[CH2,CH3]OS(=O)(=O)c1ccc(C)cc1")
            
            has_electrophile = any(
                mol.HasSubstructMatch(alkyl_halide_pattern) or mol.HasSubstructMatch(tosylate_pattern)
                for mol in reactants
            )
            
            # Check for ether formation in product (Ar-O-C)
            ether_pattern = Chem.MolFromSmarts("c[OH0]([cH0,cH1])[CH2,CH3]")
            has_ether_product = product.HasSubstructMatch(ether_pattern)
            
            # Verify C-O bond formation by checking atom mapping
            if has_phenol and has_electrophile and has_ether_product:
                return self._verify_co_bond_formation(reactants_smiles, product_smiles)
                
            return False
            
        except Exception:
            return False
    
    def _verify_co_bond_formation(self, reactants_smiles: str, product_smiles: str) -> bool:
        """Verify that a new C-O bond is formed between phenolic O and alkyl C"""
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            # Get atom mappings for oxygen and carbon that should form new bond
            product_atoms = {atom.GetAtomMapNum(): atom for atom in product.GetAtoms() if atom.GetAtomMapNum() > 0}
            
            # Find phenolic oxygen in reactants and alkyl carbon
            phenolic_o_mapnum = None
            alkyl_c_mapnum = None
            
            for reactant in reactants:
                for atom in reactant.GetAtoms():
                    if atom.GetAtomMapNum() > 0:
                        map_num = atom.GetAtomMapNum()
                        
                        # Check if this is phenolic oxygen
                        if atom.GetSymbol() == 'O' and any(
                            neighbor.GetIsAromatic() for neighbor in atom.GetNeighbors()
                        ):
                            phenolic_o_mapnum = map_num
                        
                        # Check if this is alkyl carbon next to leaving group
                        elif atom.GetSymbol() == 'C' and any(
                            neighbor.GetSymbol() in ['Br', 'Cl', 'I'] or 
                            (neighbor.GetSymbol() == 'O' and len([n for n in neighbor.GetNeighbors() if n.GetSymbol() == 'S']) > 0)
                            for neighbor in atom.GetNeighbors()
                        ):
                            alkyl_c_mapnum = map_num
            
            # Check if these atoms are bonded in the product
            if phenolic_o_mapnum and alkyl_c_mapnum and phenolic_o_mapnum in product_atoms and alkyl_c_mapnum in product_atoms:
                o_atom = product_atoms[phenolic_o_mapnum]
                c_atom = product_atoms[alkyl_c_mapnum]
                
                # Check if they are bonded in product
                return any(neighbor.GetAtomMapNum() == alkyl_c_mapnum for neighbor in o_atom.GetNeighbors())
            
            return False
            
        except Exception:
            return False
