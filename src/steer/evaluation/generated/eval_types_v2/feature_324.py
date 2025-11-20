"""Generated evaluation code for: Multi-step protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes based on protecting group cycling strategy.
    Checks if the route contains the specified number of protection/deprotection steps
    and total protecting group manipulations.
    """
    
    def __init__(self, config):
        self.target_protection_steps = config["parameters"]["protection_steps"]
        self.target_deprotection_steps = config["parameters"]["deprotection_steps"]
        self.target_total_pg_steps = config["parameters"]["total_pg_steps"]
        
        # Common protecting group patterns
        self.protection_patterns = {
            "acetate": "[CH3]C(=O)O",  # Acetyl protection
            "tips": "[Si]([CH](C)C)([CH](C)C)[CH](C)C",  # TIPS
            "mom": "COC",  # Methoxymethyl
            "tbdms": "[Si](C)(C)C(C)(C)C",  # TBDMS
            "bn": "Cc1ccccc1",  # Benzyl
            "boc": "CC(C)(C)OC(=O)",  # tert-Butoxycarbonyl
            "cbz": "O=C(OCc1ccccc1)",  # Carbobenzyloxy
            "pmb": "COc1ccc(C)cc1"  # para-Methoxybenzyl
        }
        
        # Deprotection reagent patterns
        self.deprotection_reagents = {
            "tbaf": "F[B-](F)(F)F",  # TBAF for silyl deprotection
            "tfa": "FC(F)(F)C(=O)O",  # TFA for Boc deprotection
            "pd_c": "[Pd]",  # Pd/C for hydrogenolysis
            "lialh4": "[Li+].[AlH4-]",  # LAH
            "naoh": "[Na+].[OH-]",  # Base hydrolysis
            "hcl": "[H+].[Cl-]"  # Acid hydrolysis
        }
    
    def condition_depth(self, d):
        """
        Analyzes all reactions in the route tree to count protecting group steps.
        Returns (condition_met, total_reactions).
        """
        reactions = self.get_rxns(d)
        
        protection_count = 0
        deprotection_count = 0
        
        for rxn in reactions:
            if self.is_protection_reaction(rxn):
                protection_count += 1
            elif self.is_deprotection_reaction(rxn):
                deprotection_count += 1
        
        total_pg_steps = protection_count + deprotection_count
        
        # Check if all conditions are met
        protection_match = protection_count >= self.target_protection_steps
        deprotection_match = deprotection_count >= self.target_deprotection_steps
        total_match = total_pg_steps >= self.target_total_pg_steps
        
        condition_met = protection_match and deprotection_match and total_match
        
        return condition_met, len(reactions)
    
    def is_protection_reaction(self, rxn_smiles):
        """
        Determines if a reaction is a protection reaction by checking for
        introduction of protecting groups.
        """
        try:
            reactants, products = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Check if protecting groups appear in products but not reactants
            for pg_name, pg_pattern in self.protection_patterns.items():
                pg_mol = Chem.MolFromSmarts(pg_pattern)
                if pg_mol is None:
                    continue
                
                reactant_has_pg = any(mol.HasSubstructMatch(pg_mol) for mol in reactant_mols if mol)
                product_has_pg = any(mol.HasSubstructMatch(pg_mol) for mol in product_mols if mol)
                
                # Protection: PG appears in products but not in reactants (or increases in count)
                if product_has_pg and not reactant_has_pg:
                    return True
                    
                # Check for increase in PG count
                reactant_pg_count = sum(len(mol.GetSubstructMatches(pg_mol)) for mol in reactant_mols if mol)
                product_pg_count = sum(len(mol.GetSubstructMatches(pg_mol)) for mol in product_mols if mol)
                
                if product_pg_count > reactant_pg_count:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def is_deprotection_reaction(self, rxn_smiles):
        """
        Determines if a reaction is a deprotection reaction by checking for
        removal of protecting groups or presence of deprotection reagents.
        """
        try:
            reactants, products = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Check for deprotection reagents in reactants
            for reagent_pattern in self.deprotection_reagents.values():
                reagent_mol = Chem.MolFromSmarts(reagent_pattern)
                if reagent_mol and any(mol.HasSubstructMatch(reagent_mol) for mol in reactant_mols if mol):
                    return True
            
            # Check if protecting groups disappear from reactants to products
            for pg_name, pg_pattern in self.protection_patterns.items():
                pg_mol = Chem.MolFromSmarts(pg_pattern)
                if pg_mol is None:
                    continue
                
                reactant_pg_count = sum(len(mol.GetSubstructMatches(pg_mol)) for mol in reactant_mols if mol)
                product_pg_count = sum(len(mol.GetSubstructMatches(pg_mol)) for mol in product_mols if mol)
                
                # Deprotection: PG count decreases
                if reactant_pg_count > product_pg_count:
                    return True
            
            return False
            
        except Exception:
            return False
