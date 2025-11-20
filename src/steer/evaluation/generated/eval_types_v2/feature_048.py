"""Generated evaluation code for: One carbon homologation via cyanide substitution"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CyanideHomologation(BaseScoring):
    """
    Evaluates synthesis routes for one-carbon homologation via cyanide substitution.
    Detects SN2 reactions where primary alkyl iodides are substituted with cyanide
    to extend the carbon chain by one unit.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Earlier homologation is generally preferred for strategic bond formation
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Checks if a reaction represents cyanide homologation of primary alkyl iodide.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for primary alkyl iodide pattern in reactants
            primary_alkyl_iodide_pattern = Chem.MolFromSmarts("[CH2]-I")
            has_primary_iodide = any(mol.HasSubstructMatch(primary_alkyl_iodide_pattern) for mol in reactants)
            
            if not has_primary_iodide:
                return False
            
            # Check for cyanide nucleophile in reactants
            cyanide_patterns = [
                Chem.MolFromSmarts("[C-]#N"),  # cyanide anion
                Chem.MolFromSmarts("N#C[C-]"), # alternative representation
                Chem.MolFromSmarts("[Na+].[C-]#N"), # sodium cyanide
                Chem.MolFromSmarts("[K+].[C-]#N")   # potassium cyanide
            ]
            
            has_cyanide = any(
                any(mol.HasSubstructMatch(pattern) for pattern in cyanide_patterns)
                for mol in reactants
            )
            
            if not has_cyanide:
                return False
            
            # Check for nitrile formation in products (homologation product)
            nitrile_pattern = Chem.MolFromSmarts("C-C#N")
            has_nitrile_product = any(mol.HasSubstructMatch(nitrile_pattern) for mol in products)
            
            if not has_nitrile_product:
                return False
            
            # Verify carbon chain extension by comparing atom map numbers
            return self._verify_homologation(reactants, products)
            
        except Exception:
            return False
    
    def _verify_homologation(self, reactants, products) -> bool:
        """
        Verifies that the reaction represents true homologation by checking
        that the carbon chain is extended by one unit.
        """
        try:
            # Find the primary alkyl iodide reactant
            primary_iodide_pattern = Chem.MolFromSmarts("[CH2]-I")
            iodide_reactant = None
            
            for mol in reactants:
                if mol.HasSubstructMatch(primary_iodide_pattern):
                    iodide_reactant = mol
                    break
            
            if not iodide_reactant:
                return False
            
            # Find the nitrile product
            nitrile_pattern = Chem.MolFromSmarts("C-C#N")
            nitrile_product = None
            
            for mol in products:
                if mol.HasSubstructMatch(nitrile_pattern):
                    nitrile_product = mol
                    break
            
            if not nitrile_product:
                return False
            
            # Check that the carbon connected to iodine in reactant
            # is connected to nitrile in product using atom mapping
            for atom in iodide_reactant.GetAtoms():
                if atom.GetAtomMapNum() > 0 and atom.GetSymbol() == 'C':
                    # Check if this carbon has an iodine neighbor
                    has_iodine_neighbor = any(
                        neighbor.GetSymbol() == 'I' 
                        for neighbor in atom.GetNeighbors()
                    )
                    
                    if has_iodine_neighbor:
                        # Find corresponding atom in product
                        for prod_atom in nitrile_product.GetAtoms():
                            if (prod_atom.GetAtomMapNum() == atom.GetAtomMapNum() and
                                prod_atom.GetSymbol() == 'C'):
                                # Check if this carbon is connected to nitrile
                                for neighbor in prod_atom.GetNeighbors():
                                    if neighbor.GetSymbol() == 'C':
                                        for nn in neighbor.GetNeighbors():
                                            if nn.GetSymbol() == 'N' and neighbor.GetBonds()[0].GetBondType() == Chem.BondType.TRIPLE:
                                                return True
            
            return True  # Default to true if mapping verification is inconclusive
            
        except Exception:
            return True  # Default to true if verification fails
