"""Generated evaluation code for: Oxime intermediate for amine formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class OximeIntermediateRoute(BaseScoring):
    """
    Checks if the synthesis route uses an oxime intermediate for primary amine formation.
    Detects the ketone-to-oxime-to-amine sequence rather than direct ketone reduction.
    """
    
    def __init__(self, config: Dict):
        self.oxime_pattern = Chem.MolFromSmarts("[C]=[N]-[O]")  # Oxime functional group
        self.primary_amine_pattern = Chem.MolFromSmarts("[C][NH2]")  # Primary amine
        self.ketone_pattern = Chem.MolFromSmarts("[C]=[O]")  # Ketone
    
    def route_scoring(self, x) -> float:
        """
        Score based on whether oxime intermediate route is used.
        Returns 1 if oxime route is found, 0 otherwise.
        """
        if x < 0:
            return 0  # Oxime intermediate route not found
        else:
            return 1  # Oxime intermediate route detected
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves oxime formation or reduction to primary amine.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi)
                if mol is not None:
                    reactants.append(mol)
            
            products = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi)
                if mol is not None:
                    products.append(mol)
            
            if not reactants or not products:
                return False
            
            # Check for oxime formation: ketone -> oxime
            oxime_formation = self._check_oxime_formation(reactants, products)
            
            # Check for oxime reduction: oxime -> primary amine
            oxime_reduction = self._check_oxime_reduction(reactants, products)
            
            return oxime_formation or oxime_reduction
            
        except Exception:
            return False
    
    def _check_oxime_formation(self, reactants, products) -> bool:
        """Check if reaction converts ketone to oxime"""
        # Look for ketone in reactants and oxime in products
        has_ketone_reactant = any(mol.HasSubstructMatch(self.ketone_pattern) for mol in reactants)
        has_oxime_product = any(mol.HasSubstructMatch(self.oxime_pattern) for mol in products)
        
        return has_ketone_reactant and has_oxime_product
    
    def _check_oxime_reduction(self, reactants, products) -> bool:
        """Check if reaction converts oxime to primary amine"""
        # Look for oxime in reactants and primary amine in products
        has_oxime_reactant = any(mol.HasSubstructMatch(self.oxime_pattern) for mol in reactants)
        has_amine_product = any(mol.HasSubstructMatch(self.primary_amine_pattern) for mol in products)
        
        return has_oxime_reactant and has_amine_product
