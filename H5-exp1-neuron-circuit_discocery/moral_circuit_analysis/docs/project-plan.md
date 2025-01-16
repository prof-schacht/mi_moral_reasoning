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
