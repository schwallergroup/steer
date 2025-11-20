"""Generated evaluation code for: Fmoc protecting group strategy for piperazine nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class FmocPiperazineProtection(BaseScoring):
    """
    Evaluates synthesis routes for Fmoc protecting group strategy on piperazine nitrogen.
    Checks if Fmoc protection is applied to secondary amine in piperazine and used 
    during coupling reactions.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.functional_group = config["parameters"]["functional_group"]
        self.steps_protected = config["parameters"]["steps_protected"]
        
        # Define SMARTS patterns
        self.piperazine_pattern = "[NH1]1CC[NH1]CC1"  # Piperazine with secondary amines
        self.fmoc_pattern = "C(=O)Nc1ccccc1-c2ccccc2"  # Fmoc carbamate pattern
        self.fmoc_piperazine_pattern = "[NH0](C(=O)OCC1c2ccccc2-c3ccccc31)C4CCN([H,#6])CC4"  # Fmoc-protected piperazine
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not found
        else:
            # Earlier application of protection strategy is better
            # Scale to 0-10 range where lower depth gets higher score
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves Fmoc protection of piperazine nitrogen
        or uses Fmoc-protected piperazine in a coupling reaction.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        # Parse molecules
        try:
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
                
        except:
            return False
        
        # Check for Fmoc protection reaction: piperazine + Fmoc reagent -> Fmoc-piperazine
        if self._is_fmoc_protection_reaction(react_mols, prod_mol):
            return True
            
        # Check for coupling reaction using Fmoc-protected piperazine
        if self._is_coupling_with_fmoc_piperazine(react_mols, prod_mol):
            return True
            
        return False
    
    def _is_fmoc_protection_reaction(self, reactants, product) -> bool:
        """Check if this is an Fmoc protection reaction on piperazine."""
        # Check if reactants contain piperazine and product contains Fmoc-protected piperazine
        has_piperazine_reactant = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.piperazine_pattern)) 
            for mol in reactants
        )
        
        has_fmoc_product = product.HasSubstructMatch(
            Chem.MolFromSmarts(self.fmoc_piperazine_pattern)
        )
        
        # Check for Fmoc reagent (like Fmoc-Cl or Fmoc-OSu)
        fmoc_reagent_patterns = [
            "C(=O)ClOCC1c2ccccc2-c3ccccc31",  # Fmoc-Cl
            "C(=O)ON4C(=O)CCC4=O.OCC1c2ccccc2-c3ccccc31"  # Fmoc-OSu
        ]
        
        has_fmoc_reagent = any(
            any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for pattern in fmoc_reagent_patterns)
            for mol in reactants
        )
        
        return has_piperazine_reactant and has_fmoc_product and has_fmoc_reagent
    
    def _is_coupling_with_fmoc_piperazine(self, reactants, product) -> bool:
        """Check if this is a coupling reaction using Fmoc-protected piperazine."""
        # Check if reactants contain Fmoc-protected piperazine
        has_fmoc_piperazine_reactant = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.fmoc_piperazine_pattern))
            for mol in reactants
        )
        
        # Check if this appears to be a coupling reaction (amide bond formation, etc.)
        # Look for coupling reagents or conditions
        coupling_patterns = [
            "c1ccc(P(c2ccccc2)c3ccccc3)cc1",  # Triphenylphosphine (for Mitsunobu)
            "CN(C)c1ccncc1",  # DMAP
            "CCN(CC)CC",  # Triethylamine
        ]
        
        has_coupling_reagent = any(
            any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for pattern in coupling_patterns)
            for mol in reactants
        )
        
        return has_fmoc_piperazine_reactant and (has_coupling_reagent or len(reactants) >= 2)
