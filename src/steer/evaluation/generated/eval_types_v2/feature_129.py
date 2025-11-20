"""Generated evaluation code for: Early stage Sandmeyer halogenation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SandmeyerHalogenation(BaseScoring):
    """
    Evaluates if Sandmeyer halogenation (C-N bond break converting aniline to aryl halide) 
    occurs at the desired early stage of synthesis.
    """
    
    def __init__(self, config: Dict):
        self.target_step = config["parameters"].get("step_position", 7)
        self.timing = config["parameters"].get("timing", "early")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Sandmeyer reaction doesn't occur
        
        if self.timing == "early":
            # Early stage preferred - penalize late occurrence
            return max(0, 1 - x)
        else:
            # Target specific step position
            return max(0, 1 - abs(x - self.target_step / 10.0))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Sandmeyer halogenation"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants, products = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Check for aniline pattern in reactants (aromatic amine)
            aniline_pattern = Chem.MolFromSmarts("c[NH2]")
            has_aniline = any(mol.HasSubstructMatch(aniline_pattern) for mol in reactant_mols)
            
            if not has_aniline:
                return False
            
            # Check for aryl halide formation in products
            aryl_halide_patterns = [
                Chem.MolFromSmarts("c[Cl]"),  # aryl chloride
                Chem.MolFromSmarts("c[Br]"),  # aryl bromide  
                Chem.MolFromSmarts("c[I]")    # aryl iodide
            ]
            
            has_aryl_halide = any(
                any(mol.HasSubstructMatch(pattern) for pattern in aryl_halide_patterns)
                for mol in product_mols
            )
            
            if not has_aryl_halide:
                return False
            
            # Additional check for typical Sandmeyer conditions/reagents
            # Look for copper salts or halide sources in reactants
            sandmeyer_reagents = [
                Chem.MolFromSmarts("[Cu]"),    # copper salts
                Chem.MolFromSmarts("[N+]#N"),  # diazonium
                Chem.MolFromSmarts("N#N")      # diazotization intermediate
            ]
            
            has_sandmeyer_reagent = any(
                any(mol.HasSubstructMatch(pattern) for pattern in sandmeyer_reagents)
                for mol in reactant_mols
            )
            
            # Check for C-N bond break by comparing atom mapping
            c_n_break = self._check_c_n_bond_break(reactants, products)
            
            return has_aryl_halide and (has_sandmeyer_reagent or c_n_break)
            
        except Exception:
            return False
    
    def _check_c_n_bond_break(self, reactants: str, products: str) -> bool:
        """Check if C-N bond is broken based on atom mapping"""
        try:
            reactant_mol = Chem.MolFromSmiles(reactants)
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if not reactant_mol or not all(product_mols):
                return False
            
            # Find C-N bonds in reactants with atom mapping
            for bond in reactant_mol.GetBonds():
                atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
                if ((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'N') or
                    (atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'C')):
                    
                    map1, map2 = atom1.GetAtomMapNum(), atom2.GetAtomMapNum()
                    if map1 > 0 and map2 > 0:
                        # Check if these mapped atoms are in different product molecules
                        atom1_products = []
                        atom2_products = []
                        
                        for i, prod_mol in enumerate(product_mols):
                            has_map1 = any(a.GetAtomMapNum() == map1 for a in prod_mol.GetAtoms())
                            has_map2 = any(a.GetAtomMapNum() == map2 for a in prod_mol.GetAtoms())
                            if has_map1:
                                atom1_products.append(i)
                            if has_map2:
                                atom2_products.append(i)
                        
                        # If atoms are in different products, bond was broken
                        if atom1_products and atom2_products and not set(atom1_products) & set(atom2_products):
                            return True
            
            return False
            
        except Exception:
            return False
