"""Generated evaluation code for: Evans auxiliary for achiral center formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EvansAuxiliaryAchiralCenter(BaseScoring):
    """
    Evaluates whether Evans auxiliary is used for achiral center formation.
    
    This class detects the presence of Evans oxazolidinone auxiliary in reactions
    where no new stereocenters are created in the target molecule, which represents
    an inefficient use of this chiral auxiliary.
    """
    
    def __init__(self, config: Dict):
        self.auxiliary_pattern = config["parameters"]["smarts_pattern"]
        self.expects_stereocenter = config["parameters"]["stereocenter_created"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 10  # Evans auxiliary not used inappropriately
        else:
            # Earlier use is worse (more wasteful)
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction involves Evans auxiliary for achiral center formation.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            # Parse molecules
            prod_mol = Chem.MolFromSmiles(products)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            
            if not prod_mol or not all(reactant_mols):
                return False
            
            # Check if Evans auxiliary pattern is present in reactants
            evans_pattern = Chem.MolFromSmarts(self.auxiliary_pattern)
            if not evans_pattern:
                return False
                
            has_evans_in_reactants = any(
                mol.HasSubstructMatch(evans_pattern) for mol in reactant_mols
            )
            
            if not has_evans_in_reactants:
                return False
            
            # Check if new stereocenters are created
            stereocenters_created = self._count_stereocenters_difference(
                reactant_mols, prod_mol
            )
            
            # Return True if Evans auxiliary is used but no stereocenters created
            return has_evans_in_reactants and (stereocenters_created == 0)
            
        except Exception:
            return False
    
    def _count_stereocenters_difference(self, reactants, product):
        """
        Count the difference in stereocenters between reactants and products.
        """
        try:
            # Count chiral centers in product
            product_chiral = len(Chem.FindMolChiralCenters(product, includeUnassigned=True))
            
            # Count chiral centers in all reactants
            reactants_chiral = sum(
                len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)) 
                for mol in reactants
            )
            
            return product_chiral - reactants_chiral
            
        except Exception:
            return 0
