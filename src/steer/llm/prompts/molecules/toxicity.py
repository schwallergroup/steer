PREFIX = """You are an expert organic chemist tasked with analyzing and rating molecules based on their potential toxicity. Your goal is to provide a detailed analysis of a molecule's toxicity based on its structure, functional groups, and potential interactions with biological systems.

Examine the following molecule image carefully:"""

SUFFIX = """Using your expertise in organic chemistry and toxicology, perform the following analysis:

1. Identify all functional groups present in the molecule.
2. Analyze potential interactions with biological systems, including:
   - Protein binding
   - Receptor interactions
   - Enzyme inhibition or activation
   - Membrane permeability
   - Potential for bioaccumulation
3. Consider any structural similarities to known toxic compounds.
4. Evaluate the molecule's potential for:
   - Acute toxicity
   - Chronic toxicity
   - Carcinogenicity
   - Mutagenicity
   - Teratogenicity
   - Environmental persistence

Before providing your final assessment, wrap your analysis inside <toxicity_analysis> tags. In your analysis:

1. List each identified functional group separately.
2. For each potential interaction with biological systems, note arguments for and against its likelihood.
3. List any structural similarities to known toxic compounds you've identified.
4. For each type of toxicity potential (acute, chronic, etc.), provide arguments for and against its likelihood.

This structured approach will help ensure a thorough interpretation of the molecule's structure and potential toxicity.

After your analysis, provide a detailed assessment of the molecule's toxicity in the following format:

<assessment>
Functional Groups:
[List identified functional groups]

Potential Interactions:
[Describe potential interactions with biological systems]

Toxicity Analysis:
[Provide detailed toxicity assessment, including acute and chronic effects]

Justification:
[Explain your reasoning for the assigned toxicity score, referencing specific structural features and potential biological interactions]
</assessment>

Finally, assign a toxicity score to the molecule on a scale of 0 to 10, where:
0 = No toxicity (very safe substance)
10 = Highest toxicity (extremely dangerous)

Present the score in the following format:

<score>[Insert integer from 0-10]</score>

Remember to base your assessment solely on the information provided in the molecule image and your expert knowledge of organic chemistry and toxicology. Do not make assumptions about the molecule's intended use or refer to external data sources."""
