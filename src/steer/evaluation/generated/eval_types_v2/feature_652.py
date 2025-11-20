"""Generated evaluation code for: Late stage aldehyde formation via ester reduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAldehydeFormation(BaseScoring):
    """
    Evaluates routes based on late-stage aldehyde formation via DIBAL-H reduction of methyl esters.
    Checks if the target aldehyde is installed in the final step through ester reduction.
    """
    
    def __init__(self, config: Dict):
        self.substrate_smarts = config["parameters"]["substrate_smarts"]  # "COC(=O)"
        self.product_smarts = config["parameters"]["product_smarts"]      # "C=O"
        self.stage = config["parameters"]["stage"]                       # "final"
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't happen
        
        if self.stage == "final":
            # For final stage, we want x to be close to 1.0 (happening at the end)
            if x >= 0.9:  # Final step (depth fraction close to 1)
                return 10
            elif x >= 0.7:  # Very late stage
                return 7
            elif x >= 0.5:  # Mid-late stage
                return 4
            else:  # Early stage - not desired
                return 1
        else:
            # For other stages, standard depth-based scoring
            return max(0, 10 - abs(x - 0.5) * 10)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents DIBAL-H reduction of methyl ester to aldehyde.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0].strip()
            reactants_smiles = rxn_parts[1].strip()
            
            # Parse molecules
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
                
            reactant_mols = []
            for r_smi in reactants_smiles.split("."):
                r_mol = Chem.MolFromSmiles(r_smi.strip())
                if r_mol:
                    reactant_mols.append(r_mol)
            
            if not reactant_mols:
                return False
            
            # Check if product contains aldehyde pattern
            aldehyde_pattern = Chem.MolFromSmarts(self.product_smarts)
            if not product_mol.HasSubstructMatch(aldehyde_pattern):
                return False
            
            # Check if any reactant contains methyl ester pattern
            ester_pattern = Chem.MolFromSmarts(self.substrate_smarts)
            has_ester_reactant = any(r_mol.HasSubstructMatch(ester_pattern) for r_mol in reactant_mols)
            
            if not has_ester_reactant:
                return False
            
            # Additional check: ensure the transformation makes chemical sense
            # The reactant should have ester but product should have aldehyde in corresponding position
            return self._validate_ester_to_aldehyde_transformation(reactant_mols, product_mol, mapped_rxn)
            
        except Exception:
            return False
    
    def _validate_ester_to_aldehyde_transformation(self, reactants, product, mapped_rxn):
        """
        Validate that the transformation is indeed ester reduction to aldehyde
        by checking atom mapping if available.
        """
        # Basic validation: ensure we're not just detecting coincidental substructures
        # Check that product has fewer heavy atoms than largest reactant (loss of OMe group)
        product_atoms = product.GetNumHeavyAtoms()
        max_reactant_atoms = max(r.GetNumHeavyAtoms() for r in reactants)
        
        # Ester reduction should result in loss of atoms (methoxy group becomes methanol)
        # Allow for some flexibility in case of other small molecules in reaction
        if product_atoms >= max_reactant_atoms:
            return False
            
        # Additional check: product should not contain the ester pattern
        ester_pattern = Chem.MolFromSmarts(self.substrate_smarts)
        if product.HasSubstructMatch(ester_pattern):
            return False
            
        return True
