"""Generated evaluation code for: Benzyl protecting group for phenol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylPhenolProtection(BaseScoring):
    """
    Evaluates synthesis routes for the use of benzyl protecting groups on phenols.
    Checks for benzyl ether formation (protection) and hydrogenolysis deprotection.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not found
        else:
            # Earlier protection is better (lower depth fraction)
            return 1 - x
    
    def hit_condition(self, d):
        """Check if this reaction involves benzyl protection or deprotection of phenol"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        # Check for benzyl protection (phenol + benzyl halide -> benzyl ether)
        if self._is_benzyl_protection(reactants_smiles, products_smiles):
            return True
            
        # Check for hydrogenolysis deprotection (benzyl ether -> phenol)
        if self._is_hydrogenolysis_deprotection(reactants_smiles, products_smiles):
            return True
            
        return False
    
    def _is_benzyl_protection(self, reactants_smiles, products_smiles):
        """Check if reaction is benzyl protection of phenol"""
        try:
            # Parse molecules
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Look for phenol in reactants
            phenol_pattern = Chem.MolFromSmarts("c[OH1]")
            has_phenol_reactant = any(mol.HasSubstructMatch(phenol_pattern) for mol in reactant_mols)
            
            # Look for benzyl halide/alcohol in reactants
            benzyl_halide_pattern = Chem.MolFromSmarts("c1ccccc1C[Cl,Br,I]")
            benzyl_alcohol_pattern = Chem.MolFromSmarts("c1ccccc1CO")
            has_benzyl_electrophile = any(
                mol.HasSubstructMatch(benzyl_halide_pattern) or mol.HasSubstructMatch(benzyl_alcohol_pattern)
                for mol in reactant_mols
            )
            
            # Look for benzyl ether in products
            benzyl_ether_pattern = Chem.MolFromSmarts("c1ccccc1COc2ccccc2")
            has_benzyl_ether_product = any(mol.HasSubstructMatch(benzyl_ether_pattern) for mol in product_mols)
            
            return has_phenol_reactant and has_benzyl_electrophile and has_benzyl_ether_product
            
        except:
            return False
    
    def _is_hydrogenolysis_deprotection(self, reactants_smiles, products_smiles):
        """Check if reaction is hydrogenolysis deprotection of benzyl ether"""
        try:
            # Parse molecules
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Look for benzyl ether in reactants
            benzyl_ether_pattern = Chem.MolFromSmarts("c1ccccc1COc2ccccc2")
            has_benzyl_ether_reactant = any(mol.HasSubstructMatch(benzyl_ether_pattern) for mol in reactant_mols)
            
            # Look for H2 in reactants (hydrogenolysis conditions)
            has_hydrogen = any(Chem.MolToSmiles(mol) in ["[H][H]", "[H]H"] for mol in reactant_mols)
            
            # Look for phenol in products
            phenol_pattern = Chem.MolFromSmarts("c[OH1]")
            has_phenol_product = any(mol.HasSubstructMatch(phenol_pattern) for mol in product_mols)
            
            # Look for toluene in products (from benzyl group)
            toluene_pattern = Chem.MolFromSmarts("c1ccccc1C")
            has_toluene_product = any(mol.HasSubstructMatch(toluene_pattern) for mol in product_mols)
            
            return has_benzyl_ether_reactant and (has_hydrogen or has_phenol_product) and has_toluene_product
            
        except:
            return False
