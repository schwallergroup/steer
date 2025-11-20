"""Generated evaluation code for: Schmidt glycosylation strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SchmidtGlycosylation(BaseScoring):
    """
    Evaluates presence of Schmidt glycosylation reactions using trichloroacetimidate donors.
    Checks for the characteristic trichloroacetimidate leaving group and glycosidic bond formation.
    """
    
    def __init__(self, config: Dict):
        self.donor_type = config["parameters"].get("donor_type", "trichloroacetimidate")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Schmidt glycosylation not found
        else:
            return 10 - (x * 10)  # Earlier occurrence scores higher
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents a Schmidt glycosylation"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            products = rxn_parts[0]
            reactants = rxn_parts[1].split(".")
            
            # Define trichloroacetimidate donor pattern
            # Sugar with trichloroacetimidate leaving group
            trichloroacetimidate_pattern = "[CH]([OH,O])[CH]([OH,O])[CH]([OH,O])[CH]([OH,O])[CH]([OH,O])ON=C(C(Cl)(Cl)Cl)"
            
            # Simpler pattern for trichloroacetimidate group
            imidate_pattern = "ON=C(C(Cl)(Cl)Cl)"
            
            # Pattern for glycosidic bond formation (anomeric carbon linkage)
            glycoside_pattern = "[CH]([OH,O])[CH]([OH,O])[CH]([OH,O])[CH]([OH,O])[CH]([OH,O])O[C,c]"
            
            prod_mol = Chem.MolFromSmiles(products)
            if not prod_mol:
                return False
                
            reactant_mols = []
            for r_smiles in reactants:
                r_mol = Chem.MolFromSmiles(r_smiles)
                if r_mol:
                    reactant_mols.append(r_mol)
            
            if not reactant_mols:
                return False
            
            # Check for trichloroacetimidate donor in reactants
            imidate_smart = Chem.MolFromSmarts(imidate_pattern)
            has_imidate_donor = any(r.HasSubstructMatch(imidate_smart) for r in reactant_mols)
            
            # Check for glycosidic bond in product
            glycoside_smart = Chem.MolFromSmarts(glycoside_pattern)
            has_glycoside_product = prod_mol.HasSubstructMatch(glycoside_smart)
            
            # Check that trichloroacetonitrile or related byproduct is formed
            # This indicates the imidate leaving group departed
            byproduct_pattern = "N#CC(Cl)(Cl)Cl"
            byproduct_smart = Chem.MolFromSmarts(byproduct_pattern)
            has_characteristic_byproduct = any(r.HasSubstructMatch(byproduct_smart) for r in reactant_mols)
            
            return has_imidate_donor and has_glycoside_product
            
        except Exception:
            return False
