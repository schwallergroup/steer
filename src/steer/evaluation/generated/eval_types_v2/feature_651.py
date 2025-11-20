"""Generated evaluation code for: Benzyl ester as carboxylic acid protecting group"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylEsterProtection(BaseScoring):
    """
    Evaluates synthesis routes for benzyl ester protection/deprotection strategy.
    Checks if carboxylic acids are protected as benzyl esters and later deprotected
    via hydrogenation.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group_smarts = config["parameters"]["protecting_group_smarts"]  # "C(=O)OCc1ccccc1"
        self.substrate_smarts = config["parameters"]["substrate_smarts"]  # "C(=O)O" 
        self.deprotection_method = config["parameters"]["deprotection_method"]  # "hydrogenation"
        
        # Compile SMARTS patterns
        self.benzyl_ester_pattern = Chem.MolFromSmarts(self.protecting_group_smarts)
        self.carboxylic_acid_pattern = Chem.MolFromSmarts(self.substrate_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not found
        else:
            # Earlier protection is better (closer to target molecule)
            return 1 - x
            
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves benzyl ester protection or deprotection
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Remove None molecules (failed parsing)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Check for protection: carboxylic acid -> benzyl ester
            has_carboxylic_acid_reactant = any(mol.HasSubstructMatch(self.carboxylic_acid_pattern) for mol in reactants)
            has_benzyl_ester_product = any(mol.HasSubstructMatch(self.benzyl_ester_pattern) for mol in products)
            
            # Check for deprotection: benzyl ester -> carboxylic acid (via hydrogenation)
            has_benzyl_ester_reactant = any(mol.HasSubstructMatch(self.benzyl_ester_pattern) for mol in reactants)
            has_carboxylic_acid_product = any(mol.HasSubstructMatch(self.carboxylic_acid_pattern) for mol in products)
            
            # Check if this looks like hydrogenation (presence of H2 or similar reducing conditions)
            is_hydrogenation = self._is_hydrogenation_reaction(reactants, products)
            
            # Protection step
            if has_carboxylic_acid_reactant and has_benzyl_ester_product:
                return True
                
            # Deprotection step (must be via hydrogenation)
            if has_benzyl_ester_reactant and has_carboxylic_acid_product and is_hydrogenation:
                return True
                
            return False
            
        except Exception:
            return False
            
    def _is_hydrogenation_reaction(self, reactants, products) -> bool:
        """
        Check if reaction appears to be hydrogenation based on molecular changes
        """
        # Simple heuristic: look for increase in hydrogen count or presence of H2
        reactant_smiles = [Chem.MolToSmiles(mol) for mol in reactants]
        
        # Check for explicit H2 in reactants
        if any("[H][H]" in smi or "H2" in smi for smi in reactant_smiles):
            return True
            
        # Check for overall hydrogen increase (benzyl group removal + H addition)
        total_reactant_atoms = sum(mol.GetNumAtoms() for mol in reactants)
        total_product_atoms = sum(mol.GetNumAtoms() for mol in products)
        
        # In benzyl ester deprotection, we lose benzyl group (~7 atoms) but this is rough heuristic
        if total_reactant_atoms > total_product_atoms:
            return True
            
        return False
