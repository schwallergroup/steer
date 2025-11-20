"""Generated evaluation code for: Final step nucleophilic aromatic substitution"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class FinalStepNucleophilicAromaticSubstitution(BaseScoring):
    """
    Evaluates whether the final step in the synthesis route is a nucleophilic aromatic substitution
    on a dichloroimidazopyridazine substrate with pyridylmethylamine.
    """
    
    def __init__(self, config: Dict):
        self.substrate_pattern = config.get("substrate_pattern", "n1c(Cl)c(Cl)c2nccnc21")  # dichloroimidazopyridazine
        self.nucleophile_pattern = config.get("nucleophile_pattern", "c1ccncc1CN")  # pyridylmethylamine
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        elif x == 0:
            return 10  # Perfect - occurs in final step
        else:
            return max(0, 10 - 5 * x)  # Penalize if not final step
    
    def hit_condition(self, d):
        """Check if this reaction is a nucleophilic aromatic substitution on the target substrate"""
        try:
            metadata = d.get("metadata", {})
            mapped_rxn = metadata.get("mapped_reaction_smiles", "")
            
            if not mapped_rxn or ">>" not in mapped_rxn:
                return False
                
            product_smiles, reactant_smiles = mapped_rxn.split(">>")
            reactants = reactant_smiles.split(".")
            
            # Check if product contains the substrate pattern (being transformed)
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
                
            substrate_pattern = Chem.MolFromSmarts(self.substrate_pattern)
            nucleophile_pattern = Chem.MolFromSmarts(self.nucleophile_pattern)
            
            if not substrate_pattern or not nucleophile_pattern:
                return False
            
            # Check if any reactant contains the dichloroimidazopyridazine substrate
            has_substrate = False
            has_nucleophile = False
            
            for reactant_smi in reactants:
                reactant_mol = Chem.MolFromSmiles(reactant_smi)
                if not reactant_mol:
                    continue
                    
                if reactant_mol.HasSubstructMatch(substrate_pattern):
                    has_substrate = True
                    
                if reactant_mol.HasSubstructMatch(nucleophile_pattern):
                    has_nucleophile = True
            
            # Additional check: verify this looks like SNAr by checking for Cl -> N substitution
            if has_substrate and has_nucleophile:
                return self._verify_snar_pattern(product_mol, reactants)
                
            return False
            
        except Exception:
            return False
    
    def _verify_snar_pattern(self, product_mol, reactant_smiles_list):
        """Verify that this is actually a nucleophilic aromatic substitution pattern"""
        try:
            # Look for imidazopyridazine core with amine substitution pattern
            snar_product_pattern = Chem.MolFromSmarts("n1c(NCc2ccccn2)c([Cl,N])c3nccnc31")
            if not snar_product_pattern:
                return False
                
            return product_mol.HasSubstructMatch(snar_product_pattern)
            
        except Exception:
            return False
