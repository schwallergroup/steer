"""Generated evaluation code for: Evans auxiliary protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EvansAuxiliaryStrategy(BaseScoring):
    """
    Evaluates whether an Evans auxiliary protecting group strategy is used in the synthesis route.
    An Evans auxiliary is a chiral oxazolidinone used for stereocontrol in aldol reactions,
    typically removed via H2O2/LiOH deprotection.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        
        # Evans auxiliary core structure - oxazolidinone ring
        self.evans_core = "[#6]1[#6][#7][#6](=[#8])[#8]1"
        
        # Common Evans auxiliaries with benzyl and isopropyl substituents
        self.evans_patterns = [
            # Basic oxazolidinone
            "[#6]1[#6][#7][#6](=[#8])[#8]1",
            # Benzyl Evans auxiliary
            "[#6]1[#6]([#6]c2ccccc2)[#7][#6](=[#8])[#8]1",
            # Isopropyl Evans auxiliary  
            "[#6]1[#6]([CH]([CH3])[CH3])[#7][#6](=[#8])[#8]1"
        ]
        
        # H2O2/LiOH deprotection reagents
        self.deprotection_reagents = ["[O-][O-]", "[Li+]", "[OH-]"]

    def route_scoring(self, x) -> float:
        """Convert depth fraction to 0-10 score"""
        if x < 0:
            return 0  # Evans auxiliary strategy not found
        
        if self.condition_type == "bool":
            return 10  # Found Evans auxiliary usage
        else:
            # Earlier use of Evans auxiliary is generally better for stereocontrol
            return 10 * (1 - abs(x - self.target_depth))

    def hit_condition(self, d):
        """Check if this reaction involves Evans auxiliary formation or usage"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".") if r.strip()]
            
            if not product_mol or not reactant_mols:
                return False
                
            # Check for Evans auxiliary formation (oxazolidinone appears in product)
            evans_in_product = any(product_mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                                 for pattern in self.evans_patterns)
            
            # Check for Evans auxiliary cleavage (oxazolidinone in reactants but not product)
            evans_in_reactants = any(any(rmol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                                        for pattern in self.evans_patterns)
                                   for rmol in reactant_mols if rmol)
            
            # Evans auxiliary aldol reaction (auxiliary present in both reactants and products)
            evans_aldol = evans_in_product and evans_in_reactants
            
            # Check for deprotection reagents in auxiliary cleavage
            if evans_in_reactants and not evans_in_product:
                deprotection_reagents_present = any(
                    any(rmol.HasSubstructMatch(Chem.MolFromSmarts(reagent)) 
                        for reagent in self.deprotection_reagents)
                    for rmol in reactant_mols if rmol
                )
                return deprotection_reagents_present
            
            return evans_in_product or evans_aldol
            
        except:
            return False
