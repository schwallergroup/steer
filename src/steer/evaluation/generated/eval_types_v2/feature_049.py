"""Generated evaluation code for: Malonic ester decarboxylation strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MalonicEsterDecarboxylation(BaseScoring):
    """
    Evaluates synthesis routes for malonic ester decarboxylation strategy.
    Checks for decarboxylation reactions where malonic ester substrates are converted
    to ethyl acetate products, typically following alkylation and decarboxylation sequence.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        # Malonic ester pattern: diethyl malonate or substituted versions
        self.substrate_pattern = config.get("parameters", {}).get("substrate_pattern", "C(C(=O)OCC)(C(=O)OCC)")
        # Ethyl acetate or similar monoester pattern
        self.product_pattern = config.get("parameters", {}).get("product_pattern", "CC(=O)OCC")
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0  # Strategy not found
            # Earlier use of strategy is generally better for malonic ester synthesis
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Checks if a reaction represents malonic ester decarboxylation by:
        1. Looking for CO2 loss (decarboxylation signature)
        2. Checking substrate contains malonic ester pattern
        3. Checking product contains monoester pattern
        """
        try:
            metadata = d.get("metadata", {})
            mapped_rxn = metadata.get("mapped_reaction_smiles", "")
            
            if not mapped_rxn or ">>" not in mapped_rxn:
                return False
            
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for decarboxylation signature (CO2 loss)
            has_co2_loss = self._check_decarboxylation(reactants, products)
            
            if not has_co2_loss:
                return False
            
            # Check for malonic ester substrate pattern
            substrate_mol = Chem.MolFromSmarts(self.substrate_pattern)
            has_malonic_substrate = any(reactant.HasSubstructMatch(substrate_mol) 
                                     for reactant in reactants if reactant)
            
            # Check for ester product pattern
            product_mol = Chem.MolFromSmarts(self.product_pattern)
            has_ester_product = any(product.HasSubstructMatch(product_mol) 
                                  for product in products if product)
            
            return has_malonic_substrate and has_ester_product
            
        except Exception:
            return False
    
    def _check_decarboxylation(self, reactants, products) -> bool:
        """
        Checks for CO2 loss by comparing carbon count and looking for CO2 in products.
        """
        try:
            # Count carbons in reactants vs products (excluding CO2)
            reactant_carbons = sum(mol.GetNumAtoms() for mol in reactants 
                                 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
            
            # Filter out CO2 from products for carbon counting
            non_co2_products = []
            has_co2 = False
            
            for product in products:
                if product.GetNumAtoms() == 3:  # Potential CO2
                    atoms = [atom.GetSymbol() for atom in product.GetAtoms()]
                    if atoms.count('C') == 1 and atoms.count('O') == 2:
                        has_co2 = True
                        continue
                non_co2_products.append(product)
            
            product_carbons = sum(mol.GetNumAtoms() for mol in non_co2_products
                                for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
            
            # Decarboxylation should show CO2 formation and carbon loss
            return has_co2 or (reactant_carbons > product_carbons)
            
        except Exception:
            return False
