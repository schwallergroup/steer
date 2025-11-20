"""Generated evaluation code for: Late stage ketone formation via decarboxylation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageKetoneDearboxylation(BaseScoring):
    """
    Evaluates routes that use decarboxylation of β-keto acids to form ketones at late stages.
    Checks for the transformation of [CX3](=O)[CH2][CX3](=O)[OH] to [CX3](=O)[CH3].
    """
    
    def __init__(self, config: Dict):
        self.substrate_pattern = config["parameters"]["substrate_pattern"]
        self.product_pattern = config["parameters"]["product_pattern"]
        self.substrate_mol = Chem.MolFromSmarts(self.substrate_pattern)
        self.product_mol = Chem.MolFromSmarts(self.product_pattern)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Decarboxylation doesn't happen
        else:
            # Late-stage decarboxylation is better (higher depth fraction preferred)
            return x * 10
            
    def hit_condition(self, d):
        """
        Check if this reaction represents a decarboxylation of β-keto acid to ketone.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Check if any reactant has the β-keto acid pattern
            has_substrate = any(mol.HasSubstructMatch(self.substrate_mol) for mol in reactants)
            
            # Check if any product has the ketone pattern
            has_product = any(mol.HasSubstructMatch(self.product_mol) for mol in products)
            
            # Additional check for CO2 loss (decarboxylation signature)
            has_co2_loss = self._check_co2_loss(reactants, products)
            
            return has_substrate and has_product and has_co2_loss
            
        except Exception:
            return False
            
    def _check_co2_loss(self, reactants, products):
        """
        Check if the reaction involves loss of CO2 by comparing carbon counts.
        """
        try:
            reactant_carbons = sum(len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']) 
                                 for mol in reactants)
            product_carbons = sum(len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']) 
                                for mol in products)
            
            # Decarboxylation should result in loss of one carbon
            return reactant_carbons == product_carbons + 1
            
        except Exception:
            return False
