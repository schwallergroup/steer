
prefix = """You are an expert in organic chemistry. You will be presented with a synthesis plan for a target molecule. Your task is to analyze this plan thoroughly, examining each reaction in detail and evaluating the strategic aspects of the overall route.

Here is the synthesis plan:
<synthesis_plan>
"""

suffix = """
</synthesis_plan>
Please follow these steps to analyze the synthesis plan:

1. Examine each reaction in the plan sequentially. For each reaction:
   a. Assess the feasibility of the reaction
   b. Identify potential side products
   c. Analyze the reaction mechanism
   d. Consider likely byproducts
   e. Evaluate the reaction conditions and their appropriateness
   f. Consider any potential challenges or limitations

2. Analyze the overall strategy of the synthesis route:
   a. Evaluate the efficiency of the route (number of steps, overall yield)
   b. Assess the choice of starting materials and key intermediates
   c. Identify any clever or innovative steps in the synthesis
   d. Consider alternative approaches or potential improvements
   e. Evaluate the scalability of the route
   f. Assess the route's atom economy and green chemistry principles

3. Provide your detailed analysis in <analysis> tags. Be thorough and comprehensive in your evaluation. Do not summarize or abbreviate your analysis; include all relevant details and considerations.

4. After your analysis, classify the synthesis route into one or more categories. Consider aspects such as:
   - Linear vs. convergent synthesis
   - Number of steps (e.g., short, medium, long)
   - Type of chemistry involved (e.g., transition metal-catalyzed, organocatalytic, biomimetic)
   - Overall strategy (e.g., protecting group-heavy, redox economy)
   Provide each classification in separate <class> tags.

Remember to be as detailed and thorough as possible in your analysis. Your expertise in organic chemistry should be evident in the depth and breadth of your evaluation.
"""
