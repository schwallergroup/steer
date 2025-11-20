"""Generated evaluation code for: Schmidt rearrangement carboxylic acid to amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SchmidtRearrangementDepth(BaseScoring):
    """
    Evaluates synthesis routes for Schmidt rearrangement reactions that convert
    carboxylic acids to primary amines via C-C bond breaking.
    
    The Schmidt rearrangement involves:
    1. Carboxylic acid starting material
    2. Formation of primary amine product
    3. Breaking of a C-C bond adjacent to the carboxyl group
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Schmidt rearrangement doesn't occur
        else:
            if self.condition_type == "bool":
                return 1  # Reaction found
            else:
                # Earlier occurrence is generally better for this transformation
                return max(0, 1 - x)
    
    def hit_condition(self, d):
        """
        Detects Schmidt rearrangement by checking for:
        1. Carboxylic acid in reactants
        2. Primary amine in products
        3. C-C bond breaking pattern
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            if not reactant_mols or not product_mols:
                return False
            
            # Check for carboxylic acid pattern in reactants
            carboxylic_acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
            has_carboxylic_acid = any(mol.HasSubstructMatch(carboxylic_acid_pattern) for mol in reactant_mols)
            
            # Check for primary amine pattern in products
            primary_amine_pattern = Chem.MolFromSmarts("[NX3H2][CX4]")
            has_primary_amine = any(mol.HasSubstructMatch(primary_amine_pattern) for mol in product_mols)
            
            # Basic functional group transformation check
            if not (has_carboxylic_acid and has_primary_amine):
                return False
            
            # Check for C-C bond breaking pattern characteristic of Schmidt rearrangement
            # Look for carbon atom that was connected to carboxyl carbon but is now separated
            return self._check_cc_bond_break(reactant_mols, product_mols)
            
        except Exception:
            return False
    
    def _check_cc_bond_break(self, reactants, products):
        """
        Check for characteristic C-C bond breaking in Schmidt rearrangement.
        This involves loss of carbon connectivity between the carboxyl carbon
        and adjacent carbon atoms.
        """
        try:
            # Look for mapped atoms to track bond changes
            for reactant in reactants:
                if reactant is None:
                    continue
                    
                # Find carboxylic acid carbons with atom mapping
                carboxyl_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
                matches = reactant.GetSubstructMatches(carboxyl_pattern)
                
                for match in matches:
                    carboxyl_c = match[0]  # The carboxyl carbon
                    carboxyl_atom = reactant.GetAtomWithIdx(carboxyl_c)
                    
                    if carboxyl_atom.GetAtomMapNum() == 0:
                        continue
                        
                    # Find adjacent carbons in reactant
                    adjacent_carbons = []
                    for neighbor in carboxyl_atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'C' and neighbor.GetAtomMapNum() > 0:
                            adjacent_carbons.append(neighbor.GetAtomMapNum())
                    
                    if not adjacent_carbons:
                        continue
                    
                    # Check if these carbons are separated in products
                    carboxyl_map = carboxyl_atom.GetAtomMapNum()
                    
                    for product in products:
                        if product is None:
                            continue
                            
                        carboxyl_in_product = None
                        adjacent_in_product = []
                        
                        for atom in product.GetAtoms():
                            if atom.GetAtomMapNum() == carboxyl_map:
                                carboxyl_in_product = atom
                            elif atom.GetAtomMapNum() in adjacent_carbons:
                                adjacent_in_product.append(atom)
                        
                        # If carboxyl carbon is not in this product but adjacent carbons are,
                        # or if they're in the same product but no longer bonded
                        if carboxyl_in_product is None and adjacent_in_product:
                            return True
                        elif carboxyl_in_product and adjacent_in_product:
                            # Check if they're still bonded
                            for adj_atom in adjacent_in_product:
                                bond = product.GetBondBetweenAtoms(
                                    carboxyl_in_product.GetIdx(), 
                                    adj_atom.GetIdx()
                                )
                                if bond is None:  # Bond was broken
                                    return True
            
            return False
            
        except Exception:
            return False
