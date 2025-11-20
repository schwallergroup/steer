"""Generated evaluation code for: Azide reduction with nitro group present"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AzideReductionChemoselectivity(BaseScoring):
    """
    Evaluates routes for azide reduction reactions performed in the presence of 
    nitro groups, requiring chemoselectivity to avoid reducing the nitro group.
    """
    
    def __init__(self, config: Dict):
        self.chemoselectivity_required = config["parameters"]["chemoselectivity_required"]
        self.concurrent_groups = config["parameters"]["concurrent_reducible_groups"]
        
        # SMARTS patterns for detection
        self.azide_pattern = Chem.MolFromSmarts("[N-]=[N+]=[N-]")  # Azide group
        self.nitro_pattern = Chem.MolFromSmarts("[N+](=O)[O-]")    # Nitro group
        self.amine_pattern = Chem.MolFromSmarts("[NH2,NH1,NH0]")   # Amine products
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            return 10 * (1 - x)  # Earlier is better for synthetic planning
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is an azide reduction with nitro group present"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Check if azide is consumed and amine is formed
            azide_consumed = False
            amine_formed = False
            nitro_present_reactant = False
            nitro_present_product = False
            
            # Check reactants for azide and nitro groups
            for mol in reactant_mols:
                if mol and self.azide_pattern.HasSubstructMatch(mol):
                    azide_consumed = True
                if mol and self.nitro_pattern.HasSubstructMatch(mol):
                    nitro_present_reactant = True
            
            # Check products for amine and nitro groups
            for mol in product_mols:
                if mol and self.amine_pattern.HasSubstructMatch(mol):
                    amine_formed = True
                if mol and self.nitro_pattern.HasSubstructMatch(mol):
                    nitro_present_product = True
            
            # This is an azide reduction if:
            # 1. Azide is consumed from reactants
            # 2. Amine is formed in products
            # 3. Nitro group is present in both reactants and products (preserved)
            is_azide_reduction = azide_consumed and amine_formed
            
            if self.chemoselectivity_required:
                # Chemoselectivity means nitro group must be preserved
                is_chemoselective = nitro_present_reactant and nitro_present_product
                return is_azide_reduction and is_chemoselective
            else:
                # Just check for azide reduction with nitro present
                return is_azide_reduction and nitro_present_reactant
                
        except Exception:
            return False
