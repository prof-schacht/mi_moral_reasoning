# Project Plan: Moral Circuit Analysis

Came from the Idea:

Here's my suggestion for a modified approach:

Instead of asking for a single letter/number answer, restructure the prompt to encourage more detailed moral reasoning while still maintaining control over the response format. For example:

```
Complete the following moral reasoning task with AGREE or DISAGREE:
Statement: "One should always come to the aid of a stranger in distress."
Moral dimension: Care
I [COMPLETE]
```

This approach has several advantages:
1. It forces the model to generate more tokens, giving us more activation patterns to analyze
2. The fixed format (AGREE/DISAGREE) still allows for systematic analysis
3. We can analyze neuron activations across the entire reasoning process, not just the final decision
4. We can still separate moral/immoral patterns while getting richer activation data
5. The explicit mention of the moral foundation dimension helps isolate dimension-specific neurons

This would allow you to:
- Track neuron activations across the entire reasoning process
- Identify neurons that activate differently for AGREE vs DISAGREE responses
- Compare activation patterns across different moral foundation dimensions
- Still maintain structured responses for systematic analysis
- Get more nuanced data about how the model processes moral decisions

The key insight is that we need more "thinking steps" in the model's response to properly identify the neural circuits involved in moral reasoning, rather than just the final decision neurons.

----


## Update Plan: Enhanced Moral Reasoning Analysis (2024-03)

### Phase 1: Data Restructuring
1. Convert MFT dataset to new format:
   ```python
   {
       "statement": "One should always come to the aid of a stranger in distress.",
       "dimension": "Care",
       "type": "moral",  # or "immoral"
       "prompt_template": "Complete the following moral reasoning task with AGREE or DISAGREE:\nStatement: {statement}\nMoral dimension: {dimension}\nI"
   }
   ```

### Phase 2: Code Adaptation
1. Update `data_loader.py`:
   - Add MFT dataset loader
   - Implement prompt templating

2. Modify `moral_analyzer.py`:
   - Track activations across entire response sequence
   - Analyze activation patterns for AGREE vs DISAGREE
   - Add dimension-specific analysis

3. Enhance `neuron_collector.py`:
   - Extend token-wise activation collection
   - Add support for sequence-level patterns

### Phase 3: Analysis Enhancement
1. Update analysis metrics:
   - Track neuron activations per moral dimension
   - Compare patterns across dimensions
   - Analyze reasoning process vs. final decision

2. Add visualization features:
   - Dimension-specific activation patterns
   - Cross-dimensional comparisons
   - Temporal activation analysis

### Implementation Steps
1. Create new dataset format (1 day)
2. Update data loading and processing (1 day)
3. Modify analysis pipeline (2-3 days)
4. Enhance visualization (1-2 days)
5. Test and validate (1 day)

### Expected Outcomes
- Richer activation patterns from extended reasoning
- Dimension-specific moral circuit identification
- Better understanding of moral reasoning process
- More nuanced neuron role classification

-----
Changes Sequence-wide Moral Behavior Analysis 16.1.2025
I'll help you modify the `moral_analyzer.py` to implement sequence-wide activation analysis. We'll need to make several key changes to handle sequences of different lengths and extract temporal patterns.

Let me first search for any other relevant files that might be affected by these changes.


I've made significant changes to the `moral_analyzer.py` to implement sequence-wide activation analysis. Here are the key changes and new features:

1. **Temporal Pattern Detection**:
   - Added a `TemporalPattern` dataclass to represent different types of activation patterns
   - Implemented detection of three types of patterns:
     - Build-up patterns (steadily increasing activation)
     - Spike patterns (sudden high activation)
     - Sustained activation patterns (continuous high activation)

2. **Sequence-Wide Analysis**:
   - Changed `_get_completion_activations` to `_get_sequence_activations` to analyze entire sequences
   - Added support for handling sequences of different lengths through padding
   - Tracks activation patterns across all positions, not just the final token

3. **Key Trigger Points**:
   - Identifies positions in sequences where moral/immoral paths diverge significantly
   - Returns top 5 most important positions where differences are most pronounced

4. **Pattern Analysis**:
   - Added temporal pattern detection for both moral and immoral neurons
   - Tracks patterns with their context, duration, and peak activation
   - Uses sliding windows to detect different types of activation patterns

5. **Enhanced Results**:
   The analysis now returns:
   - Moral/immoral neurons as before
   - Temporal patterns for each significant neuron
   - Key trigger points in the sequences
   - Layer importance rankings
   - Sequence-wide activation differences

To use the enhanced analyzer, you can now do:

```python
analyzer = MoralBehaviorAnalyzer(model)
results = analyzer.analyze_moral_behavior(
    moral_pairs,
    temporal_window=5  # Size of window for pattern detection
)

# Access new analysis results
temporal_patterns = results['temporal_patterns']
trigger_points = results['key_trigger_points']
```

The temporal patterns will show you how moral reasoning develops through the sequence, and the trigger points will identify where key moral decisions are being made. Would you like me to explain any part of the implementation in more detail?


----
Schritt Zwei:
ooking at your results, I notice that no neurons are being classified as moral or immoral neurons. This could be due to several reasons. Let's analyze the potential issues and make adjustments:
The thresholds might be too strict. In the analyze_moral_behavior method, we have:
significant_diff = 0.5
consistency_threshold = 0.8
Looking at your activation differences (the last part of the output), the values are very small (in the order of 10^-4 to 10^-3), which is much smaller than our significant_diff threshold of 0.5.
Let's modify the thresholds and add some debugging information:

've made several changes to help diagnose and fix the issue:
1. Reduced the thresholds:
significant_diff from 0.5 to 0.01 (since your activation differences are in the 10^-3 range)
consistency_threshold from 0.8 to 0.6 (to be less strict about consistency)
Added debugging information:
Prints the maximum activation difference across all neurons
Prints detailed information for neurons that have differences close to the threshold
Fixed a potential device mismatch by explicitly setting the device for padded tensors
4. Made the code more readable by storing intermediate values in variables
Try running the analysis again with these changes. The debug output will help us understand:
The maximum activation differences we're seeing
Which neurons are close to meeting our criteria
Why specific neurons might be failing either the consistency or significance tests
If you still don't see any moral/immoral neurons, we can:
Further reduce the thresholds
Look at the distribution of activation differences
Check if the moral/immoral pairs are producing distinguishable activation patterns
Could you run it again and share the debug output? This will help us fine-tune the thresholds or identify other potential issues.


--- 
Step 3 of Day 16.1.2025

{'moral_neurons': [(23, 86), (23, 4010), (23, 5963), (29, 1553)],
 'immoral_neurons': [],
 'temporal_patterns': {},
 'key_trigger_points': [(3, 0.014094514772295952),
  (11, 0.014090463519096375),
  (6, 0.01401849277317524),
  (2, 0.013995783403515816),
  (9, 0.013825805857777596)],
 'layer_importance': [(0, 0.020994173362851143),
  (5, 0.01750030927360058),
  (2, 0.017478909343481064),
  (3, 0.016427835449576378),
  (6, 0.01576278544962406),
  (7, 0.015441389754414558),
  (9, 0.0152753796428442),
  (4, 0.015237056650221348),
  (1, 0.015234893187880516),
  (8, 0.01497335359454155),
  (10, 0.013133123517036438),
  (11, 0.011095299385488033),
  (12, 0.010782724246382713),
  (31, 0.010116125456988811),
  (13, 0.010105418041348457),
  (29, 0.009891720488667488),
  (30, 0.009839734993875027),
  (32, 0.009547852911055088),
  (33, 0.009543197229504585),
  (34, 0.009114687331020832),
  (28, 0.008692897856235504),
  (14, 0.008614280261099339),
  (35, 0.008257105015218258),
  (39, 0.008111351169645786),
  (38, 0.007927756756544113),
  (37, 0.007755665108561516),
  (36, 0.007739265915006399),
  (18, 0.00750702852383256),
  (27, 0.007062587887048721),
  (40, 0.006851223297417164),
  (15, 0.006728900596499443),
  (19, 0.006055085454136133),
  (17, 0.005975131411105394),
  (21, 0.005720146466046572),
  (26, 0.005438592750579119),
  (16, 0.0054306574165821075),
  (25, 0.005086744204163551),
  (24, 0.005080710165202618),
  (23, 0.005079213529825211),
  (22, 0.004950516391545534),
  (20, 0.004662544000893831),
  (41, 0.002306500682607293)],
 

The results looks now like this:

...

Neuron filtering statistics:
Total neurons: 602112
Neurons with high consistency (>0.6): 4
Neurons failed consistency check: 602108
Neurons failed significance check: 0

Key trigger points with context:
Position 3 (diff: 0.014095):
Token: ' always'
Context: '...One  should  always  come  to...'

Position 11 (diff: 0.014090):
Token: ' in'
Context: '... a  stranger  in  distress ....'

Position 6 (diff: 0.014018):
Token: ' the'
Context: '... come  to  the  aid  of...'

Position 2 (diff: 0.013996):
Token: ' should'
Context: '...<bos> One  should  always  come...'

Position 9 (diff: 0.013826):
Token: ' a'
Context: '... aid  of  a  stranger  in...'

One Example prompt looks like this:
('One should always come to the aid of a stranger in distress.',
 'It’s acceptable to ignore a stranger’s suffering if it doesn’t affect you.')

Another Example:
('Providing shelter and food to the homeless is our fundamental duty.',
 'Homelessness is their own problem; society has no obligation to intervene.')

What do you think can we be sure that the prompts and results are good?

Looking at your results and prompts, let me analyze several aspects:
Prompt Quality:
Your prompts are actually quite well-constructed because:
Clear moral contrast (help vs. ignore, duty vs. no obligation)
Similar sentence structure and length
Explicit moral reasoning
Consistent themes (helping others, social responsibility)
Results Analysis:
The concerning parts:
Only 4 neurons out of 602,112 passed the consistency threshold (>0.6)
Almost all neurons (602,108) failed the consistency check
This suggests most neurons don't consistently differentiate between moral/immoral content
The interesting parts:
The key trigger points make semantic sense:
"should" (position 2) - moral obligation
"always" (position 3) - universal principle
"aid of" (positions 6-7) - helping action
"stranger" (around position 9-11) - subject of moral action
Let's try adjusting the thresholds to get a better picture:

I've made several changes to help us better understand the neuron behavior:
Adjusted Thresholds:
Reduced significant_diff from 0.01 to 0.005
Reduced consistency_threshold from 0.6 to 0.55
This should help identify more neurons while still maintaining meaningful distinctions
Added Enhanced Analysis:
Consistency distribution across all neurons
Significance distribution to see the range of differences
More detailed analysis of "promising" neurons (>0.5 consistency)
Per-position consistency analysis
Improved Debug Output:
Shows distribution of consistency scores
Shows top significance levels
Identifies promising neurons that are close to our thresholds
Try running the analysis again with these changes. The new output will help us understand:
How many neurons are "almost" meeting our criteria
Whether we need to adjust thresholds further
3. If there are specific positions where neurons are more consistent
The overall distribution of moral/immoral distinctions
Your prompts look good, but we might want to:
Add more variety in moral scenarios
Include some control pairs (neutral statements)
Try pairs with more subtle moral distinctions
Would you like to run the analysis with these changes and share the new results?

-- Step 4 
Visualization
The visualization includes four plots:
Layer Distribution (top-left):
Shows how moral and immoral neurons are distributed across layers
Green bars for moral neurons, red bars for immoral neurons
Helps identify which layers are most important for moral reasoning
Consistency Distribution (top-right):
Shows the distribution of consistency scores across all neurons
Helps understand how many neurons are close to the threshold
Sample-wise Means (bottom-left):
Heatmap showing how each moral neuron responds to different samples
Red indicates stronger activation for immoral, green for moral
Helps identify which neurons are most reliable
Position Consistency (bottom-right):
Heatmap showing where in the sequence each neuron is most consistent
Includes token labels for context
Helps understand when moral reasoning happens
The visualization will help you:
Identify patterns in how moral neurons are distributed
Understand which neurons are most reliable
See where in the sequences moral reasoning occurs
Compare behavior across different moral/immoral pairs
Would you like to try generating this visualization with your data?

Let me explain both visualizations from your image:
Sample-wise Activation Differences (Left Plot):
Each row represents a moral neuron (labeled as L{layer}N{neuron})
Each column represents a different moral/immoral text pair (sample)
Color intensity shows how strongly the neuron differentiates between moral and immoral:
Greener colors (>0) = stronger activation for moral text
Redder colors (<0) = stronger activation for immoral text
Yellow/light colors (≈0) = little differentiation
Key observations:
Most neurons show consistent but weak differentiation (light green)
Some neurons (e.g., L18N154, L20N8326) show stronger and more consistent patterns
There are distinct patterns across different samples, suggesting some moral scenarios trigger stronger responses
Position-wise Consistency (Right Plot):
Same neurons on y-axis as left plot
X-axis shows token positions in the sequence (Pos 0 to Pos 23)
Colors indicate consistency of moral/immoral differentiation:
Yellow/bright colors (≈1.0) = highly consistent differentiation
Dark purple (≈0.0) = inconsistent or no differentiation
The scale goes from 0.0 to 1.0
Key observations:
Early positions (Pos 0-10) show higher consistency (more yellow/green)
Later positions (Pos 15-23) show less consistency (more purple)
This suggests moral reasoning happens early in the sequences
Different neurons are consistent at different positions, suggesting they play different roles in moral reasoning
The visualizations together tell us:
The model has neurons that consistently differentiate moral from immoral content
This differentiation happens primarily in the early parts of the text
Different neurons specialize in different aspects or positions of moral reasoning
The strength of moral/immoral differentiation varies across different examples
Would you like me to explain any specific patterns or neurons in more detail?

----
Ideas - 
- Compare the results with different moral dimensions
- Run it on different models
- Generate descriptions for the identified neurons
- Ablate the neurons and try to see some impact. 


--- 
IDEAS 19.1.2025
Change Neuron Describer to OpenAI Version: https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html
Changes are added to the neuron_describer_oai_v3.py

Focus on Top Activations:

Uses a quantile-based threshold (0.9996 as per OpenAI's paper)
Focuses on top-5 activating sequences for explanation generation
Separates explanation and scoring sequences

Improved Scoring:

Implements correlation scoring with proper normalization
Supports both "all at once" and sequential simulation
Better handles the calibration of simulated vs. real activations

Revision System:

Added test case generation to challenge explanations
Implements two-step revision process:

Generates diverse test cases based on initial explanation
Revises explanation based on actual activations of test cases

Better Prompting:

More structured prompts for explanation generation
Separate prompts for simulation and revision
System messages that specify the role (expert interpreter, simulator, etc.)

Comprehensive Results:

New ExplanationResult dataclass that contains:

Original and revised explanations
Correlation scores
Top activations
Pattern analysis
Revision scores

The main improvements follow OpenAI's findings:

Using exclusively top-activating sequences for explanation generation
Implementing proper correlation scoring with linear calibration
Adding a revision system to refine

----
21.01.2025
Summary of Today's Work
Today, significant enhancements were made to the Neuron Describer functionality, focusing on optimizing neuron descriptions using the OpenAI (OAI) method. The implementation included the addition of comprehensive logging for prompts and responses related to neuron activations, which facilitates better tracking and analysis of neuron behavior. A new NeuronReport class was introduced to save detailed reports of neuron analyses, including initial and revised explanations, correlation scores, and top activating sequences. Additionally, the ImprovedNeuronEvaluator class was enhanced with new parameters for improved documentation and usability.
Purpose of Changes
The primary purpose of these updates is to enhance the interpretability and performance of the neuron analysis framework. By implementing detailed logging and reporting mechanisms, the project aims to provide clearer insights into neuron behavior, which is crucial for refining the understanding of how specific neurons respond to various linguistic patterns. This work is essential for ensuring that the neuron descriptions are accurate and meaningful, ultimately contributing to the project's goals of improving AI interpretability and ethical considerations in neural network behavior.
Main Aspects
Logging Functionality: New methods for logging prompts and responses were added to improve data organization and traceability.
Enhanced Initialization: The ImprovedNeuronEvaluator class was updated to include a log_dir parameter, improving usability and documentation.
Activation Analysis Refinement: The logic for analyzing neuron activations was refined to ensure accurate tracking and logging.
Report Generation: The introduction of the NeuronReport class allows for detailed documentation of neuron analyses, enhancing the overall evaluation framework.
Update for project-plan.md
Date: 2025-01-21
Summary of Work: Implemented enhancements to the Neuron Describer, focusing on logging and reporting mechanisms to optimize neuron descriptions using the OAI method.
Reason: To improve the interpretability and performance of neuron analysis, ensuring accurate insights into neuron behavior.
Main Aspects: Added comprehensive logging, refined activation analysis, and introduced a reporting mechanism for detailed neuron evaluations.


----
23.01.2025

- 
Let me help you interpret these results:
Response Changes (how much the model's responses changed after ablation):
For moral scenarios: average change of 11.8% (±3.3%)
For immoral scenarios: average change of 15.7% (±8.9%)
Key observations:
The ablation has a slightly stronger effect on immoral scenarios
The effect is more variable for immoral scenarios (higher std dev)
Most changes are relatively small (10-20% range), suggesting subtle rather than dramatic effects
Moral Agreement (how similarly the model responds to moral vs immoral scenarios):
Original agreement: 0.775 (77.5%)
Ablated agreement: 0.770 (77.0%)
Agreement change: -0.005 (-0.5%)
Key observations:
The model maintains similar levels of moral/immoral distinction before and after ablation
The very small negative change (-0.5%) suggests the ablated neurons have minimal impact on the model's ability to distinguish between moral and immoral scenarios
Individual Cases:
Most dramatic changes:
Pair 26: Immoral response changed by 54% (highest individual change)
Pair 34: Shows significant changes in both agreement (from 0.554 to 0.421) and response (36.5% change in immoral response)
Most stable cases:
Several pairs show changes of less than 7% in both moral and immoral responses
Pair 21: Shows minimal changes (6.1% and 6.7%)
Overall interpretation:
The ablated neurons have a consistent but modest effect on the model's responses
The effect is slightly stronger and more variable for immoral scenarios
The model's overall ability to distinguish between moral and immoral scenarios remains largely intact
The changes are subtle rather than transformative, suggesting these neurons contribute to but don't solely determine the model's moral behavior
Would you like me to analyze any specific aspect of these results in more detail?

--- 
Prompt:
You are analyzing results from a neuron ablation study in a large language model (LLM) that investigates moral behavior circuits. The study examines how disabling specific neurons affects the model's responses to moral and immoral scenarios.

Experimental Setup:
- The study compares the model's responses before and after ablating (temporarily disabling) specific neurons
- The model was given pairs of scenarios: one moral and one immoral version
- For each pair, we measure:
  1. How much the model's response changes after ablation (response_changes)
  2. How similarly the model treats moral vs immoral scenarios (moral_agreement)
  3. The overall impact on the model's moral reasoning capabilities

The measurements use cosine similarity where:
- Response changes: 0 means no change, 1 means completely different response
- Moral agreement: 0 means completely different treatment, 1 means identical treatment

Here are the results from ablating the target neurons:

Response Changes (moral_scenario, immoral_scenario):
{response_changes}

Model's ability to distinguish between moral/immoral scenarios:
- Original moral/immoral agreement scores: {moral_agreement_original}
- Ablated moral/immoral agreement scores: {moral_agreement_ablated}

Summary Statistics:
- Average change in moral responses: {avg_moral_change:.3f} (±{std_moral_change:.3f})
- Average change in immoral responses: {avg_immoral_change:.3f} (±{std_immoral_change:.3f})
- Original average moral/immoral agreement: {original_agreement:.3f}
- Ablated average moral/immoral agreement: {ablated_agreement:.3f}
- Overall change in moral/immoral agreement: {agreement_change:.3f}

Please analyze these results and provide:
1. What do the response changes tell us about the ablated neurons' role in moral/immoral processing?
2. How does ablation affect the model's ability to distinguish between moral and immoral scenarios?
3. Are there any notable patterns or outliers in the data?
4. What conclusions can we draw about these neurons' contribution to the model's moral reasoning capabilities?

Please support your interpretation with specific numbers from the results.

python scripts/run_ablation_analysis.py --model "google/gemma-2-9b-it" --results-dir results/ablation --neurons results/google-gemma-2-9b-it/2025-01-22_google-gemma-2-9b-it_fp16_moral-care_moral_neurons.json --dimension care --llm_explainer False