"""Generated evaluation code for: Boc protecting group for amine selectivity"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocAmineProtection(BaseScoring):
    """
    Evaluates synthesis routes based on the use of Boc protecting groups for amine selectivity.
    
    This scoring function identifies reactions where Boc (tert-butyloxycarbonyl) groups are used
    to protect amines, enabling selective transformations at other functional groups.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
        
        # SMARTS patterns for Boc group and protected amine
        self.boc_pattern = Chem.MolFromSmarts("[CX3](=O)OC(C)(C)C")  # Boc carbonyl
        self.boc_amine_pattern = Chem.MolFromSmarts("NC(=O)OC(C)(C)C")  # Boc-protected amine
        
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10 scale)"""
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0  # Protection strategy not found
            # Earlier protection is generally better for selectivity
            return 1 - x
            
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves Boc protection of an amine for selectivity purposes.
        
        Returns True if:
        1. Boc group is introduced in this step (protection reaction)
        2. An amine is being protected with Boc
        3. The reaction appears to be for selectivity (other reactive groups present)
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol:
                    reactant_mols.append(mol)
                    
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol:
                    product_mols.append(mol)
                    
            if not reactant_mols or not product_mols:
                return False
                
            # Check if Boc group is introduced (more Boc-protected amines in products)
            reactant_boc_count = sum(len(mol.GetSubstructMatches(self.boc_amine_pattern)) 
                                   for mol in reactant_mols)
            product_boc_count = sum(len(mol.GetSubstructMatches(self.boc_amine_pattern)) 
                                  for mol in product_mols)
            
            boc_introduced = product_boc_count > reactant_boc_count
            
            if not boc_introduced:
                return False
                
            # Check if this appears to be for selectivity by looking for other reactive groups
            # that could compete with the amine
            selectivity_indicators = [
                Chem.MolFromSmarts("[OH1]"),  # Free hydroxyl groups
                Chem.MolFromSmarts("C=C"),    # Alkenes
                Chem.MolFromSmarts("c1ccccc1[OH1]"),  # Phenols
                Chem.MolFromSmarts("[SH1]"),  # Thiols
                Chem.MolFromSmarts("C(=O)[OH1]"),  # Carboxylic acids
            ]
            
            # Check if products have other reactive functional groups that would benefit from amine protection
            has_competing_groups = False
            for mol in product_mols:
                for pattern in selectivity_indicators:
                    if mol.HasSubstructMatch(pattern):
                        has_competing_groups = True
                        break
                if has_competing_groups:
                    break
                    
            return has_competing_groups
            
        except Exception:
            return False
