"""Generated evaluation code for: Late stage complex double cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageComplexDoubleCyclization(BaseScoring):
    """
    Evaluates routes for late-stage complex double cyclization reactions.
    Checks if the final step forms exactly 2 rings simultaneously through
    intramolecular cyclization.
    """
    
    def __init__(self, config: Dict):
        self.rings_formed = config["parameters"]["rings_formed"]
        self.step_position = config["parameters"]["step_position"]
        self.cyclization_type = config["parameters"]["cyclization_type"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        
        if self.step_position == "final":
            # For final step, we want x to be close to 1.0 (very late stage)
            if x > 0.9:  # Final 10% of the route
                return 10
            elif x > 0.8:  # Final 20% of the route
                return 7
            else:
                return 3  # Too early in the route
        else:
            # General late-stage preference
            return 10 * x  # Later is better
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction step forms exactly 2 rings simultaneously"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Count rings in product
            product_rings = len(Chem.GetSymmSSSR(product))
            
            # Count rings in reactants
            reactant_rings = sum(len(Chem.GetSymmSSSR(r)) for r in reactants)
            
            # Check if exactly the target number of rings were formed
            rings_formed = product_rings - reactant_rings
            
            if rings_formed != self.rings_formed:
                return False
            
            # Check if it's intramolecular (single reactant with multiple reactive sites)
            if self.cyclization_type == "intramolecular":
                # Look for a single main reactant that could undergo double cyclization
                # Filter out small molecules (likely reagents)
                main_reactants = [r for r in reactants if r.GetNumAtoms() > 5]
                
                if len(main_reactants) != 1:
                    return False
                
                main_reactant = main_reactants[0]
                
                # Check that the main reactant has multiple potential cyclization sites
                # Look for atoms that could participate in ring formation
                cyclizable_atoms = 0
                for atom in main_reactant.GetAtoms():
                    # Count heteroatoms and carbonyls as potential cyclization points
                    if (atom.GetSymbol() in ['N', 'O', 'S'] or 
                        any(bond.GetBondType() == Chem.BondType.DOUBLE and 
                            bond.GetOtherAtom(atom).GetSymbol() == 'O' 
                            for bond in atom.GetBonds())):
                        cyclizable_atoms += 1
                
                # Need at least 4 potential cyclization points for double cyclization
                return cyclizable_atoms >= 4
            
            return True
            
        except Exception:
            return False
