"""Generated evaluation code for: Bulky trityl protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BulkyTritylProtectingGroup(BaseScoring):
    """
    Evaluates the use of bulky trityl protecting group strategy for selective 
    primary alcohol protection. Checks for bis(p-anisyl)phenylmethyl ether 
    protection pattern and scores based on when this bulky selective protection occurs.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.3)
        
        # SMARTS pattern for bis(p-anisyl)phenylmethyl ether protecting group
        # This represents the bulky trityl-like structure with two p-anisyl groups
        self.protecting_group_smarts = "[CH2]-[O]-[CH](-c1ccc(OC)cc1)-c2ccc(OC)cc2"
        self.trityl_pattern = Chem.MolFromSmarts(self.protecting_group_smarts)
        
        # Pattern for primary alcohol that gets protected
        self.primary_alcohol_smarts = "[CH2]-[OH]"
        self.primary_alcohol_pattern = Chem.MolFromSmarts(self.primary_alcohol_smarts)

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not found
        
        if self.condition_type == "bool":
            return 10 if x >= 0 else 0
        else:
            # Early protection is preferred for protecting group strategies
            if x <= self.target_depth:
                return 10
            else:
                # Penalize late protection
                penalty = (x - self.target_depth) * 15
                return max(0, 10 - penalty)

    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves the formation of bis(p-anisyl)phenylmethyl ether
        protecting group on a primary alcohol.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(products)
            if not product_mol:
                return False
                
            # Check if product contains the protecting group pattern
            has_protecting_group = product_mol.HasSubstructMatch(self.trityl_pattern)
            
            if not has_protecting_group:
                return False
            
            # Check if reactants contain primary alcohol that's being protected
            reactant_mols = []
            for reactant_smiles in reactants.split("."):
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol:
                    reactant_mols.append(mol)
            
            # Look for primary alcohol in reactants that's absent in product
            has_primary_alcohol_reactant = any(
                mol.HasSubstructMatch(self.primary_alcohol_pattern) 
                for mol in reactant_mols
            )
            
            # Verify this is a protection reaction (primary alcohol converted to ether)
            if has_primary_alcohol_reactant and has_protecting_group:
                # Additional check: ensure the reaction involves bulky protection
                # Look for the trityl-type reagent in reactants
                trityl_reagent_pattern = Chem.MolFromSmarts("c1ccc(OC)cc1-[CH](-c2ccc(OC)cc2)-[Cl,Br,I]")
                has_trityl_reagent = any(
                    mol.HasSubstructMatch(trityl_reagent_pattern) if mol else False
                    for mol in reactant_mols
                )
                
                return has_trityl_reagent
                
        except Exception:
            return False
            
        return False
