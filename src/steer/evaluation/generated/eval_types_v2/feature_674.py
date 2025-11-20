"""Generated evaluation code for: Temporary N-sulfinyl protection of ketone"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class NSulfinylKetoneProtection(BaseScoring):
    """
    Evaluates synthesis routes for temporary N-sulfinyl protection of ketones.
    Looks for the formation of N-sulfinylimine from ketone followed by deprotection
    to generate a free imine intermediate.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not found
        else:
            # Earlier use of protection is generally better
            return max(0, 1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves N-sulfinyl protection of a ketone.
        Look for ketone -> N-sulfinylimine conversion.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants_smiles, products_smiles = mapped_rxn.split(">>")
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Check for ketone in reactants
            ketone_pattern = Chem.MolFromSmarts("[C,c]C(=O)[C,c]")  # Ketone pattern
            has_ketone_reactant = any(mol.HasSubstructMatch(ketone_pattern) for mol in reactant_mols)
            
            # Check for N-sulfinylimine in products
            # N-sulfinylimine pattern: C=N-S(=O)-R
            nsulfinylimine_pattern = Chem.MolFromSmarts("C=N-S(=O)")
            has_nsulfinylimine_product = any(mol.HasSubstructMatch(nsulfinylimine_pattern) for mol in product_mols)
            
            # Check for sulfinamide reagent in reactants (typical N-sulfinyl source)
            sulfinamide_pattern = Chem.MolFromSmarts("N-S(=O)-C")
            has_sulfinamide_reactant = any(mol.HasSubstructMatch(sulfinamide_pattern) for mol in reactant_mols)
            
            # Condition met if ketone + sulfinamide -> N-sulfinylimine
            if has_ketone_reactant and has_sulfinamide_reactant and has_nsulfinylimine_product:
                return True
                
            # Also check for the reverse reaction (deprotection)
            # N-sulfinylimine -> imine + sulfinate
            has_nsulfinylimine_reactant = any(mol.HasSubstructMatch(nsulfinylimine_pattern) for mol in reactant_mols)
            imine_pattern = Chem.MolFromSmarts("C=N")
            has_imine_product = any(mol.HasSubstructMatch(imine_pattern) for mol in product_mols)
            
            # Check if imine product doesn't have sulfinyl group (deprotected)
            if has_imine_product:
                for mol in product_mols:
                    if mol.HasSubstructMatch(imine_pattern) and not mol.HasSubstructMatch(nsulfinylimine_pattern):
                        if has_nsulfinylimine_reactant:
                            return True
            
            return False
            
        except Exception:
            return False
