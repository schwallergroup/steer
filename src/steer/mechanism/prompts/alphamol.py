prefix = """You are an expert chemist tasked with evaluating proposed reaction mechanisms. Your goal is to analyze each elementary step of a proposed mechanism and determine how well it aligns with the target reaction. 

Here is the target reaction you should keep in mind:
<target_reaction>"""

intermed = """</target_reaction>
The mechanistic sequence of steps you will evaluate are usually incomplete, e.g. they not always connect to the product. However you should evaluate their potential to explain the target reaction, upon completion. This will help us identify if this is a promising direction or not.
Now, here are the proposed steps you need to evaluate:
<proposed_mechanism>
"""

suffix = """</proposed_mechanism>

Analyze the proposed mechanism based on your chemistry knowledge. Consider the following aspects:
1. Does each step follow established chemical principles?
2. Are the intermediates and transition states reasonable?
3. Does the mechanism account for all reactants and products in the target reaction?
4. Is the proposed mechanism consistent with known reaction conditions (if specified)?
5. Are there any missing steps or unnecessary steps?


Provide your analysis inside <analysis> tags. Be thorough and specific in your evaluation.

After your analysis, assign a score between 0 and 10 to the proposed mechanism, where 0 indicates a completely incorrect or implausible mechanism, and 10 indicates a perfect mechanism that fully explains the target reaction. Provide a brief justification for your score.

Format your response as follows:
<analysis>
[Your detailed analysis here]
</analysis>

<score_justification>
[Brief justification for your score]
</score_justification>

<score>
[Your integer score between 0 and 10]
</score>

Remember to base your evaluation solely on the information provided in the target reaction and proposed mechanism. Do not introduce additional assumptions or information not present in the input."""
