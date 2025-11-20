"""Generated evaluation code for: Late halogen exchange bromide to iodide"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateHalogenExchange(BaseScoring):
    """
    Evaluates if a halogen exchange reaction (Br to I) occurs at the final step.
    Checks for copper-catalyzed Finkelstein reaction converting aryl bromide to aryl iodide.
    """
    
    def __init__(self, config: Dict):
        self.from_halogen = config["parameters"]["from_halogen"]
        self.to_halogen = config["parameters"]["to_halogen"]
        self.timing = config["parameters"]["timing"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        
        if self.timing == "final_step":
            # For final step, we want x to be close to 1.0 (last reaction)
            if x >= 0.9:  # Final step (allowing small tolerance)
                return 10
            else:
                return 0  # Not in final step
        else:
            # For other timing preferences, score based on depth
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents a halogen exchange from Br to I"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Check for halogen exchange pattern
            return self._detect_halogen_exchange(reactant_mols, product_mols)
            
        except Exception:
            return False
    
    def _detect_halogen_exchange(self, reactants, products):
        """Detect if halogen exchange from Br to I occurred"""
        
        # Count halogens in reactants and products
        reactant_br_count = sum(self._count_halogen(mol, "Br") for mol in reactants)
        reactant_i_count = sum(self._count_halogen(mol, "I") for mol in reactants)
        
        product_br_count = sum(self._count_halogen(mol, "Br") for mol in products)
        product_i_count = sum(self._count_halogen(mol, "I") for mol in products)
        
        # Check if Br decreased and I increased (indicating Br->I exchange)
        br_decreased = reactant_br_count > product_br_count
        i_increased = product_i_count > reactant_i_count
        
        # The change should be equal (one Br lost = one I gained)
        br_change = reactant_br_count - product_br_count
        i_change = product_i_count - reactant_i_count
        
        # Verify this is a Br to I exchange with equal stoichiometry
        if br_decreased and i_increased and br_change == i_change and br_change > 0:
            # Additional check: look for iodide source in reactants (like KI, NaI)
            has_iodide_source = any(self._has_iodide_salt(mol) for mol in reactants)
            return has_iodide_source
        
        return False
    
    def _count_halogen(self, mol, halogen_symbol):
        """Count specific halogen atoms in molecule"""
        if mol is None:
            return 0
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == halogen_symbol:
                count += 1
        return count
    
    def _has_iodide_salt(self, mol):
        """Check if molecule contains iodide (like in KI, NaI salts)"""
        if mol is None:
            return False
        
        # Look for iodide patterns common in Finkelstein reactions
        iodide_patterns = [
            "[I-]",  # Iodide ion
            "[K+].[I-]",  # KI
            "[Na+].[I-]",  # NaI
            "I"  # Simple iodine
        ]
        
        mol_smiles = Chem.MolToSmiles(mol)
        
        # Check for iodide salts or simple iodide presence
        has_iodide = any(self._count_halogen(mol, "I") > 0 for _ in [mol])
        
        # Additional check for typical iodide sources
        typical_sources = ["KI", "NaI", "[I-]"]
        has_typical_source = any(source in mol_smiles for source in typical_sources)
        
        return has_iodide or has_typical_source
