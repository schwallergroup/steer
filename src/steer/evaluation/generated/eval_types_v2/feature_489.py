"""Generated evaluation code for: Strategic silyl-to-iodide conversion for regioselective halogenation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SilylToIodideConversion(BaseScoring):
    """
    Evaluates synthesis routes for strategic silyl-to-iodide conversion reactions.
    Detects electrophilic desilylation where TMS groups are converted to iodides
    for regioselective halogenation purposes.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        
        # SMARTS patterns for TMS groups and iodides
        self.tms_pattern = "[Si](C)(C)C"  # Trimethylsilyl group
        self.iodide_pattern = "[c,C][I]"  # Aromatic or aliphatic carbon-iodine bond
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Conversion doesn't happen
        else:
            if self.condition_type == "bool":
                return 1  # Conversion found
            else:
                # Earlier conversion is better for strategic purposes
                return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Checks if a reaction involves conversion of TMS to iodide.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse reactants and products
            reactants = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi.strip())
                if mol:
                    reactants.append(mol)
            
            products = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi.strip())
                if mol:
                    products.append(mol)
            
            if not reactants or not products:
                return False
            
            # Check for TMS in reactants and iodide in products
            has_tms_reactant = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.tms_pattern))
                for mol in reactants
            )
            
            has_iodide_product = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.iodide_pattern))
                for mol in products
            )
            
            # Verify it's a desilylation by checking TMS loss
            if has_tms_reactant and has_iodide_product:
                # Additional check: ensure TMS is lost (not present in products)
                has_tms_product = any(
                    mol.HasSubstructMatch(Chem.MolFromSmarts(self.tms_pattern))
                    for mol in products
                )
                
                # True conversion: TMS in reactants, iodide in products, no TMS in products
                return not has_tms_product
            
            return False
            
        except Exception:
            return False
