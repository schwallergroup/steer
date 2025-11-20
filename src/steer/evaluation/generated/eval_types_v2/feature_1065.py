"""Generated evaluation code for: Late stage amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Evaluates synthesis routes for late-stage amide coupling reactions.
    Scores routes higher when amide bond formation occurs in the final steps,
    indicating a convergent synthetic approach.
    """
    
    def __init__(self, config: Dict):
        self.step_position = config.get("step_position", "final")
        self.timing_preference = config.get("timing", "late")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No amide coupling found
        
        if self.step_position == "final":
            # Heavily favor final step (depth 0)
            if x == 0:
                return 10
            elif x <= 0.2:  # Very late stage
                return 8
            elif x <= 0.4:  # Late stage
                return 6
            else:
                return 2  # Found but not late stage
        else:
            # General late-stage preference
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d):
        """
        Detects amide coupling reactions by looking for:
        1. Amide bond formation (C(=O)N pattern)
        2. Common amide coupling reagents/conditions
        3. Reactants with carboxylic acid/ester and amine functionalities
        """
        metadata = d.get("metadata", {})
        
        # Check reaction SMILES for amide formation
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        if not mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            reactants_smiles = rxn_parts[0]
            product_smiles = rxn_parts[1]
            
            # Parse molecules
            reactant_mols = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi)
                if mol:
                    reactant_mols.append(mol)
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
            
            # Check for amide bond formation
            if self._is_amide_coupling(reactant_mols, product_mol):
                return True
            
            # Check policy name for amide coupling indicators
            policy_name = metadata.get("policy_name", "").lower()
            amide_keywords = ["amide", "coupling", "peptide", "dcc", "edc", "hatu", "tbtu"]
            if any(keyword in policy_name for keyword in amide_keywords):
                return True
            
        except Exception:
            pass
        
        return False
    
    def _is_amide_coupling(self, reactants, product):
        """
        Determines if the reaction represents amide bond formation
        by checking for carboxylic acid/ester + amine -> amide transformation
        """
        # Amide pattern in product
        amide_pattern = Chem.MolFromSmarts("[C](=[O])[N]")
        if not product.HasSubstructMatch(amide_pattern):
            return False
        
        # Look for carboxylic acid/ester and amine in reactants
        carboxylic_pattern = Chem.MolFromSmarts("[C](=[O])[O]")  # COOH or COOEt
        ester_pattern = Chem.MolFromSmarts("[C](=[O])[O][C]")     # COOEt
        amine_pattern = Chem.MolFromSmarts("[N;!$(N=*);!$(N#*)]") # Primary or secondary amine
        
        has_carbonyl_source = False
        has_amine = False
        
        for mol in reactants:
            if mol.HasSubstructMatch(carboxylic_pattern) or mol.HasSubstructMatch(ester_pattern):
                has_carbonyl_source = True
            if mol.HasSubstructMatch(amine_pattern):
                has_amine = True
        
        return has_carbonyl_source and has_amine
