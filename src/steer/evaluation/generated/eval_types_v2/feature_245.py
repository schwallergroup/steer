"""Generated evaluation code for: Late stage quinoline-pyridine coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageQuinolinePyridineCoupling(BaseScoring):
    """
    Evaluates whether a quinoline-pyridine C-N coupling occurs at late stage.
    Detects formation of C-N bond between quinoline and pyridine rings via
    coupling reactions like Buchwald-Hartwig amination.
    """
    
    def __init__(self, config):
        self.bond_smarts = config["parameters"]["bond_smarts"]
        self.timing = config["parameters"]["timing"]
        self.reaction_type = config["parameters"]["reaction_type"]
        
        # Create RDKit pattern for the quinoline-pyridine C-N bond
        self.bond_pattern = Chem.MolFromSmarts(self.bond_smarts)
        
        # Patterns for quinoline and pyridine fragments after bond breaking
        self.quinoline_pattern = Chem.MolFromSmarts("[#6]1:[#6]:[#6]:[#7]:[#6]:[#6]:1[NH2,NH]")
        self.pyridine_pattern = Chem.MolFromSmarts("[#7]1:[#6]:[#6]:[#6]:[#6]:[#6]:1[Cl,Br,I]")

    def route_scoring(self, x):
        if x < 0:
            return 0  # Bond formation doesn't happen
        else:
            # For late-stage coupling, higher depth fraction is better
            # Scale to 0-10 where 10 is latest possible
            return min(10, x * 10)

    def hit_condition(self, d):
        """Check if this reaction involves quinoline-pyridine coupling formation"""
        metadata = d.get("metadata", {})
        
        # Check if it's a coupling reaction type
        if "policy_name" in metadata:
            policy = metadata["policy_name"].lower()
            if "coupling" not in policy and "amination" not in policy:
                return False
        
        # Get mapped reaction SMILES
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        reactants_smiles, product_smiles = rxn_smiles.split(">>")
        
        product_mol = Chem.MolFromSmiles(product_smiles)
        if not product_mol:
            return False
            
        # Check if product contains the quinoline-pyridine C-N bond
        if not product_mol.HasSubstructMatch(self.bond_pattern):
            return False
            
        # Parse reactants
        reactant_smiles_list = reactants_smiles.split(".")
        reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactant_smiles_list]
        reactant_mols = [mol for mol in reactant_mols if mol is not None]
        
        # Check if reactants contain quinoline and pyridine fragments separately
        has_quinoline_fragment = False
        has_pyridine_fragment = False
        
        for reactant in reactant_mols:
            if reactant.HasSubstructMatch(self.quinoline_pattern):
                has_quinoline_fragment = True
            if reactant.HasSubstructMatch(self.pyridine_pattern):
                has_pyridine_fragment = True
                
        # Verify this is a coupling reaction: separate fragments in reactants,
        # coupled product, and no quinoline-pyridine bond in reactants
        reactants_combined = Chem.MolFromSmiles(reactants_smiles.replace(".", ""))
        has_bond_in_reactants = False
        if reactants_combined:
            has_bond_in_reactants = reactants_combined.HasSubstructMatch(self.bond_pattern)
            
        return (has_quinoline_fragment and has_pyridine_fragment and 
                not has_bond_in_reactants)
