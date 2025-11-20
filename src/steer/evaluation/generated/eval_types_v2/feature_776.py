"""Generated evaluation code for: Beta-keto acid esterification under decarboxylation conditions"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BetaKetoAcidEsterification(BaseScoring):
    """
    Evaluates routes for beta-keto acid esterification reactions that pose decarboxylation risk.
    
    Detects esterification reactions involving beta-keto acids, which are thermally sensitive
    and prone to decarboxylation side reactions during the esterification process.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 1.0)
        
        # Beta-keto acid pattern: keto group beta to carboxylic acid
        self.beta_keto_acid_pattern = Chem.MolFromSmarts("[C](=[O])-[C]-[C](=[O])-[OH]")
        # Ester pattern for products
        self.ester_pattern = Chem.MolFromSmarts("[C](=[O])-[O]-[C]")
        
    def route_scoring(self, x) -> float:
        """Convert depth fraction to penalty score (0-10 scale)"""
        if x < 0:
            return 0  # Reaction not found
        
        if self.condition_type == "bool":
            return 10  # High penalty for risky reaction presence
        else:
            # Earlier occurrence (lower depth) gets higher penalty
            penalty = max(0, 10 - (x * 10))
            return penalty
    
    def hit_condition(self, d) -> bool:
        """Check if reaction involves beta-keto acid esterification"""
        try:
            metadata = d.get("metadata", {})
            rxn_smiles = metadata.get("mapped_reaction_smiles")
            
            if not rxn_smiles:
                return False
            
            # Parse reaction SMILES
            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                return False
                
            reactants_smiles = parts[0]
            products_smiles = parts[1]
            
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    reactant_mols.append(mol)
            
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    product_mols.append(mol)
            
            if not reactant_mols or not product_mols:
                return False
            
            # Check for beta-keto acid in reactants
            has_beta_keto_acid = any(
                mol.HasSubstructMatch(self.beta_keto_acid_pattern) 
                for mol in reactant_mols
            )
            
            if not has_beta_keto_acid:
                return False
            
            # Check for ester formation (ester in products but not in reactants)
            reactant_has_ester = any(
                mol.HasSubstructMatch(self.ester_pattern) 
                for mol in reactant_mols
            )
            
            product_has_ester = any(
                mol.HasSubstructMatch(self.ester_pattern) 
                for mol in product_mols
            )
            
            # Esterification: ester formed in products that wasn't in reactants
            is_esterification = product_has_ester and not reactant_has_ester
            
            return is_esterification
            
        except Exception:
            return False
