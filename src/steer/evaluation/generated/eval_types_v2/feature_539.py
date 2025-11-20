"""Generated evaluation code for: Commercial reagent synthesis within main route"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CommercialReagentSynthesis(MultiRxnCondBase):
    """
    Checks for multi-step synthesis of commercially available reagents within the main synthetic route.
    Penalizes routes that synthesize reagents like tributyltin hydride and 2,3-butanediol 
    when they are commercially available.
    """
    
    def __init__(self, config):
        self.step_count_threshold = config.get("step_count_threshold", 3)
        self.commercial_availability = config.get("commercial_availability", True)
        
        # Common commercially available reagents that shouldn't be synthesized
        self.commercial_reagents = [
            "CCCC[Sn](CCCC)(CCCC)[H]",  # tributyltin hydride
            "C[CH](O)[CH](O)C",         # 2,3-butanediol
            "CC(O)C(O)C",               # 2,3-butanediol isomer
            "[H][Sn](CCCC)(CCCC)CCCC",  # tributyltin hydride alt
            "O[BH3-]",                  # borane complexes
            "CC(C)CC(C)(C)O[AlH2]",     # DIBAL-H
            "C[Si](C)(C)Cl",            # TMSCl
            "CC(C)N(C(C)C)C(C)C",       # diisopropylamine
        ]
    
    def condition_depth(self, d):
        """
        Checks if the route contains multi-step synthesis of commercially available reagents.
        Returns (condition_met, reagent_synthesis_steps)
        """
        reactions = self.get_rxns(d)
        reagent_synthesis_steps = 0
        condition_met = False
        
        # Track potential reagent synthesis sequences
        reagent_sequences = self.identify_reagent_sequences(reactions)
        
        for sequence in reagent_sequences:
            if len(sequence) >= self.step_count_threshold:
                reagent_synthesis_steps += len(sequence)
                condition_met = True
        
        return condition_met, reagent_synthesis_steps
    
    def identify_reagent_sequences(self, reactions):
        """
        Identifies sequences of reactions that synthesize commercially available reagents.
        """
        sequences = []
        current_sequence = []
        
        for rxn in reactions:
            if self.is_reagent_synthesis_step(rxn):
                current_sequence.append(rxn)
            else:
                if len(current_sequence) > 0:
                    # Check if sequence produces commercial reagent
                    if self.sequence_produces_commercial_reagent(current_sequence):
                        sequences.append(current_sequence)
                    current_sequence = []
        
        # Check final sequence
        if len(current_sequence) > 0 and self.sequence_produces_commercial_reagent(current_sequence):
            sequences.append(current_sequence)
            
        return sequences
    
    def is_reagent_synthesis_step(self, rxn):
        """
        Determines if a reaction step is part of reagent synthesis rather than main product synthesis.
        """
        rxn_smiles = rxn.split(">>")
        if len(rxn_smiles) != 2:
            return False
            
        products = rxn_smiles[1].split(".")
        reactants = rxn_smiles[0].split(".")
        
        # Check for organometallic reagent formation
        organometallic_patterns = [
            "[Sn]",  # tin compounds
            "[Al]",  # aluminum compounds  
            "[B]",   # boron compounds
            "[Li]",  # lithium compounds
            "[Mg]",  # magnesium compounds
        ]
        
        for product in products:
            try:
                mol = Chem.MolFromSmiles(product)
                if mol:
                    for pattern in organometallic_patterns:
                        if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                            return True
            except:
                continue
                
        # Check for simple alcohol/diol synthesis
        diol_pattern = Chem.MolFromSmarts("[CH]([OH])[CH]([OH])")
        for product in products:
            try:
                mol = Chem.MolFromSmiles(product)
                if mol and mol.HasSubstructMatch(diol_pattern):
                    return True
            except:
                continue
                
        return False
    
    def sequence_produces_commercial_reagent(self, sequence):
        """
        Checks if a reaction sequence produces a commercially available reagent.
        """
        if not sequence:
            return False
            
        final_rxn = sequence[-1]
        rxn_smiles = final_rxn.split(">>")
        if len(rxn_smiles) != 2:
            return False
            
        products = rxn_smiles[1].split(".")
        
        for product in products:
            try:
                product_mol = Chem.MolFromSmiles(product)
                if not product_mol:
                    continue
                    
                for commercial_smiles in self.commercial_reagents:
                    try:
                        commercial_mol = Chem.MolFromSmiles(commercial_smiles)
                        if commercial_mol and product_mol.HasSubstructMatch(commercial_mol):
                            return True
                    except:
                        continue
            except:
                continue
                
        return False
