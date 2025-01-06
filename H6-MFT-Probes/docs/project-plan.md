# Moral Foundations Theory Detection in Language Models: A Probing Study

## Project Overview
This project aims to investigate how Language Models (LLMs) internally represent and process moral reasoning across different layers of their architecture. By utilizing the Moral Foundations Reddit Comments (MFRC) dataset and probing techniques, we seek to understand where and how moral foundation categories are encoded within the model's residual stream.

## Research Goals
1. Map the representation of Moral Foundations Theory (MFT) categories across different layers of LLMs
2. Identify specific layers or layer clusters where moral reasoning emerges
3. Track the evolution of moral foundation detection through the model's depth
4. Compare representation patterns across different model architectures
5. Investigate potential correlations between model behavior and human moral intuitions

## Technical Architecture

### Data Pipeline
1. Dataset Processing
   - Load MFRC dataset from Hugging Face
   - Clean and preprocess text data
   - Encode MFT categories
   - Create efficient data loading mechanisms
   - Implement comprehensive label distribution logging
   - Handle edge cases for underrepresented moral foundation categories
   - Configure logging system for detailed process tracking
   - Handle single-label and multi-label cases separately
   - Filter dataset to focus on single-label instances
   - Track and log multi-label combinations for future analysis
   - Implement bidirectional label mapping (label↔index)
   - Robust error handling for label mapping
   - Filter dataset for Everyday Morality bucket
   - Process single-label cases within Everyday Morality
   - Centralized utility functions for dataset operations
   - Reusable dataset initialization across different scripts

2. Feature Extraction
   - Hook into LLM's residual stream
   - Extract activations from all layers
   - Implement caching mechanisms for efficient processing
   - Handle batch processing for large-scale analysis

3. Probe Architecture
   - Linear probes for each layer
   - Training infrastructure with proper validation
   - Metrics collection and logging
   - Cross-validation setup

### Implementation Steps Original Plan

#### Phase 1: Data Preparation (Current)
- [x] Dataset loading and preprocessing
- [x] Efficient data structures for text and labels
- [x] Batch processing setup
- [x] Label distribution analysis and logging
- [x] Handling of underrepresented moral foundation categories
- [x] Fallback mechanisms for data splitting
- [x] Logging system configuration and implementation
- [x] Single-label vs multi-label handling
- [x] Dataset filtering and statistics logging
- [x] Bidirectional label mapping implementation
- [x] Label mapping error handling
- [x] Dataset filtering for Everyday Morality bucket
- [x] Utility functions for dataset operations
- [x] Centralized dataset initialization

#### Phase 2: LLM Integration (Current)
- [x] Model configuration and initialization system
- [x] NNSight integration for activation extraction
- [x] Flexible model switching support
- [x] Activation extraction pipeline
- [x] Multi-architecture support (GPT, Gemma, etc.)
- [x] Activation storing over all layers is to big 5 GB per Batch. We must focus on some of the layers. Perhaps every 5th layer And only the activations of the last token.
- [x] Activation storage and management

#### Phase 3: Probe Implementation (Future)
- [ ] Linear probe architecture
- [ ] Training loop implementation
- [ ] Cross-validation framework
- [ ] Metrics collection system

#### Phase 4: Analysis Tools (Future)
- [ ] Visualization of layer-wise performance
- [ ] Statistical analysis of probe results
- [ ] Comparative analysis across models
- [ ] Interactive exploration tools

## Technical Requirements

### Core Libraries
- `torch`: Neural network implementation
- `transformers`: LLM model access and manipulation
- `datasets`: Efficient dataset handling
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `sklearn`: Machine learning utilities
- `wandb` (recommended): Experiment tracking
- `matplotlib/plotly`: Visualization
- `nnsights`: Mechanistic interpretability library used for getting activation and patching of models.
- `logging`: Standard Python logging for process tracking and debugging



## Future Extensions
1. Multi-model Analysis
   - Compare representations across different LLM architectures
   - Investigate size scaling effects
   - Cross-architecture patterns

2. Advanced Probing Techniques
   - Non-linear probes
   - Attention-based probes
   - Causal intervention studies

3. Interpretation Tools
   - Interactive visualization dashboard
   - Probe analysis toolkit
   - Case study generation system

4. Dataset Extensions
   - Additional moral reasoning datasets
   - Cross-domain validation
   - Synthetic data generation

5. Multi-label Analysis (New)
   - Extend pipeline to handle multi-label cases
   - Implement multi-label classification
   - Compare single-label vs multi-label representations
   - Investigate moral foundation co-occurrence patterns

## Project Structure
```
mft_probing/
├── data/
│   ├── preprocessing/
│   ├── caching/
│   └── validation/
├── models/
│   ├── llm_hooks/
│   ├── probes/
│   └── training/
├── analysis/
│   ├── visualization/
│   ├── statistics/
│   └── interpretation/
├── configs/
├── notebooks/
```

## Current Status
The initial data processing pipeline has been implemented with a focus on single-label instances from the Everyday Morality bucket. Multi-label cases are currently excluded but logged for potential future analysis. The pipeline includes comprehensive label distribution analysis and handles edge cases appropriately. The next phase will focus on LLM activation extraction and probe training infrastructure.

## Research Questions to Address
1. At what layers do different moral foundations emerge?
2. How do representation patterns differ across foundation categories?
3. Are there clear boundaries between moral and non-moral reasoning in LLMs?
4. How do model size and architecture affect moral reasoning representation?
5. Can we identify specific attention patterns associated with moral reasoning?


----
Chronological changes - Every change should be added as a new step with a new number.
Step 20: 
------
The key changes are:
Added model type detection
Support for different model architectures (GPT and Gemma initially)
Dynamic layer name generation based on model type
Better error handling and logging
Updated project plan to reflect multi-architecture support
This should now work with the Gemma model's architecture. The extractor will automatically detect the model type and use the appropriate layer names and structure.

Step 21:
------
Fixed activation extraction issues:
- Removed attention_mask from model invocation to prevent input conflicts
- Added proper checkpoint saving with error handling
- Added tracking of successful batch processing
- Improved error handling and recovery
- Added batch processing statistics
- Implemented safe checkpoint saving mechanism

Step 22:
------
Fixed NNSight integration issues:
- Changed model invocation from forward() to trace() for better compatibility
- Removed explicit input_ids passing to prevent input conflicts
- Updated activation extraction to use NNSight's tracing mechanism
- Added better error handling for model tracing

Step 23:
------
Fixed layer access in activation extraction:
- Replaced unsafe eval() with proper attribute access
- Added safe layer access method using getattr
- Improved error handling for layer access
- Added granular logging for layer access issues
- Added better error recovery for individual layer failures

Step 24:
------
Fixed NNSight input preparation:
- Added proper input preparation for NNSight model tracing
- Included both input_ids and attention_mask in prepared inputs
- Updated trace() call to use prepared inputs
- Improved error handling for input preparation
- Added logging for input preparation process

Step 25:
------
Fixed NNSight tracing implementation:
- Updated model tracing to use nested context managers (trace and invoke)
- Removed prepared_inputs in favor of direct input_ids passing
- Simplified input handling to match NNSight examples
- Added proper context management for tracing
- Improved error handling for tracing context

Step 26:
------
Fixed NNSight activation extraction:
- Added proper token indexing using .t[0] for activation extraction
- Added handling for tuple outputs from model
- Improved value extraction from NNSight saved values
- Added proper error handling for different value types
- Updated activation processing to handle model output format changes

Step 27:
------
Fixed NNSight layer access and model output handling:
- Added proper model execution before accessing layers
- Updated logits extraction to handle model output format
- Fixed layer initialization issue in trace context
- Improved handling of model output types
- Added proper error handling for empty layer access

Step 28:
------
Fixed NNSight implementation:
- Corrected usage of NNSight's tracing pattern
- Properly nested trace() and invoke() contexts
- Removed incorrect runner() call
- Improved activation saving mechanism
- Added proper tensor type checking

The difficulty with NNSight implementation stems from:
1. Different paradigm: NNSight uses a specific tracing/intervention pattern that's different from regular PyTorch
2. Context management: Requires proper nesting of trace() and invoke() contexts
3. Value access: Saved values need special handling after the trace completes
4. Model architecture differences: Different models (GPT vs Gemma) require different access patterns
5. Documentation interpretation: The examples in the docs need careful adaptation for our use case

Step 30:
------
Simplified memory optimization:
- Added immediate CPU offloading of activations
- Implemented per-batch saving
- Removed activation accumulation
- Added basic memory cleanup
- Simplified the extraction process

Key changes:
1. Move activations to CPU immediately after extraction
2. Save each batch's results immediately
3. Clear memory after each batch
4. Remove unnecessary data accumulation

Step 31:
------
Fixed NNSight value access timing:
- Separated activation saving and value access
- Wait for trace context to complete before accessing values
- Added proper handling of saved items
- Improved error handling for value access
- Added clear separation between saving and processing phases

Key changes:
1. Store saved items during trace without accessing values
2. Process values only after trace context completes
3. Improved error handling for value access timing
4. Clear separation of saving and processing phases

Step 32:
------
Optimized layer extraction:
- Added layer stride parameter to control extraction frequency
- Modified layer selection to extract every 5th layer
- Reduced memory usage by extracting fewer layers
- Added configuration option for layer extraction frequency
- Updated initialization to support configurable layer stride

Key changes:
1. Extract every 5th layer instead of all layers
2. Added layer_stride parameter for flexibility
3. Reduced memory footprint
4. Maintained key layer coverage while reducing data size

Step 33:
------
Added NNSight exploration tools:
- Created scratchpad for exploring model structure
- Added hookpoint discovery functionality
- Implemented activation shape analysis
- Added model architecture inspection
- Created documentation of available intervention points

Key additions:
1. Model structure visualization
2. Available hookpoint discovery
3. Activation shape analysis
4. Architecture-specific exploration
5. Memory-efficient model inspection

Step 34:
------
Improved batch format handling:
- Added support for multiple batch formats
- Handle both tuple/list and dictionary batch structures
- More flexible input extraction
- Better error handling for batch processing
- Improved logging of batch structure

Key changes:
1. Support for tuple/list batches
2. Support for dictionary batches
3. Flexible input extraction
4. Enhanced batch processing robustness

Step 35:
------
Enhanced batch structure handling:
- Added detailed batch structure logging
- Improved error messages for batch processing
- Support for multiple text field names
- Better debugging information
- Robust error handling with detailed feedback

Key changes:
1. Added batch structure logging
2. Support for multiple text field names
3. Improved error messages
4. Enhanced debugging capabilities

Step 36:
------
Added support for pre-tokenized inputs:
- Handle both raw text and pre-tokenized inputs
- Detect input format automatically
- Skip unnecessary tokenization
- Improved input handling efficiency
- Better memory usage for pre-tokenized data

Key changes:
1. Support for pre-tokenized inputs
2. Automatic input format detection
3. Skip redundant tokenization
4. Enhanced memory efficiency

Step 37:
------
Improved activation storage strategy:
- Consolidated activations by layer instead of by batch
- Single file per layer per split (train/val/test)
- More efficient for probe training
- Better organization of activation data
- Reduced file system overhead

Key changes:
1. Layer-wise activation accumulation
2. Single file per layer
3. Improved data organization
4. Optimized for probe training
5. Progress tracking during extraction

Step 38:
------
Enhanced memory management for activation extraction:
- Implemented temporary file storage for batch activations
- Memory-efficient layer-wise combination
- Progressive cleanup of temporary files
- Reduced peak memory usage
- Added safeguards for cleanup

Key changes:
1. Temporary storage of batch activations
2. Progressive memory cleanup
3. Layer-wise combination strategy
4. Automatic cleanup of temporary files
5. Improved error handling with cleanup

Step 39:
------
Added layer analysis capabilities:
- Layer dimension analysis
- Memory requirement calculation
- Activation statistics computation
- Sparsity pattern analysis
- Example activation storage
- Metadata tracking for activations

Key additions:
1. Layer property analysis
2. Memory footprint calculation
3. Statistical analysis tools
4. Activation visualization prep
5. Metadata management

Step 40:
------
Enhanced layer extraction strategy:
- Always include last layer in extraction
- Extract every nth layer plus final layer
- Better coverage of model's processing stages
- Guaranteed access to final representations
- Improved logging of extracted layers

Key changes:
1. Last layer always included
2. Maintain layer extraction stride
3. Sorted layer indices
4. Better layer selection logging
5. Complete model coverage

Step 41:
------
Implemented probe training:
- Linear probes for each layer
- 8-class classification (MFT categories)
- Training with validation-based early stopping
- Comprehensive evaluation metrics
- Efficient probe storage

Key features:
1. Layer-wise probe training
2. Best model selection based on validation
3. Detailed performance metrics
4. Memory-efficient implementation
5. Automated training for all layers

### Phase 2: Model Integration
- [x] NNSight integration for activation extraction
- [x] Memory-efficient activation storage
- [x] Layer selection strategy (every nth layer)
- [x] Probe training implementation
- [ ] Evaluation metrics implementation

## Updated Project Plan

### Core Components

- Data Pipeline
  - [x] MFRC dataset loading and preprocessing
  - [x] Label distribution analysis and logging
  - [x] Handling of underrepresented categories
  - [x] Memory-optimized activation extraction (last token only)
  - [x] Efficient data saving and memory management

### Phase 1: Data Preparation
- [x] Implement data loading pipeline
- [x] Add comprehensive logging
- [x] Handle edge cases in data processing
- [x] Optimize memory usage in activation extraction
  - [x] Extract only last token activations
  - [x] Implement immediate CPU offloading
  - [x] Add layer-wise memory clearing

### Phase 2: Model Integration
- [x] NNSight integration for activation extraction
- [x] Memory-efficient activation storage
- [x] Layer selection strategy (every nth layer)
- [x] Basic probe training implementation
- [x] Enhanced probe architecture and training
- [ ] Comprehensive evaluation metrics

### Phase 3: Analysis
- [ ] Probe performance evaluation
- [ ] Visualization of results
- [ ] Statistical analysis
- [ ] Documentation of findings

### Current Status
- Implemented memory-optimized activation extraction
  - Reduced memory usage by extracting only last token activations
  - Added immediate CPU offloading and memory clearing
  - Implemented layer-wise processing to manage memory
- Improved data processing pipeline
  - Added robust error handling
  - Enhanced logging for better debugging
  - Optimized storage strategy

### Next Steps
1. Monitor and analyze probe performance across layers
   - Track performance patterns across model depth
   - Identify layers with strongest MFT representations
   - Compare early vs late layer performance

2. Implement comprehensive evaluation metrics
   - Per-class performance analysis
   - Confusion matrix visualization
   - Statistical significance testing
   - Cross-validation for robustness

3. Investigate performance bottlenecks
   - Analyze class imbalance impact
   - Study activation patterns
   - Evaluate architecture choices
   - Test alternative training strategies

4. Begin analysis phase
   - Visualize probe performance across layers
   - Study correlation with model architecture
   - Analyze relationship with MFT categories
   - Document findings and insights

### Core Libraries
- PyTorch
- NNSight
- NumPy
- Pandas
- scikit-learn
- logging
- transformers

### Notes
- Memory optimization achieved by:
  - Extracting only last token activations
  - Immediate CPU offloading
  - Layer-wise memory clearing
  - Efficient storage strategy

- Probe training challenges:
  - Class imbalance handling
  - Activation scaling
  - Architecture optimization
  - Training dynamics

- Current focus:
  - Improving probe performance
  - Handling class imbalance
  - Optimizing training process
  - Comprehensive evaluation

Step 42:
------
Enhanced probe architecture and training:
- Improved probe architecture with hidden layers
- Added class imbalance handling
- Enhanced training dynamics
- Better monitoring and logging

Key changes:
1. Probe Architecture:
   - Layer normalization for activation scaling
   - Hidden layer (256 dim) with GELU activation
   - Dropout (0.2) for regularization
   - Two-layer architecture for better representation learning

2. Class Imbalance Handling:
   - Dynamic class weight computation
   - Weighted CrossEntropyLoss
   - Class distribution logging
   - Weight clamping to prevent instability

3. Training Improvements:
   - Increased batch size (128)
   - ReduceLROnPlateau scheduler for adaptive learning rate
   - Early stopping with small improvement threshold
   - Increased training epochs and patience
   - AdamW optimizer with weight decay
   - Learning rate reduction on plateau
   - Validation-based scheduler updates

4. Monitoring Enhancements:
   - Detailed class-wise metrics
   - Label distribution tracking
   - Training dynamics logging
   - GPU utilization monitoring

Findings and Observations:
- Initial probe performance showed heavy class bias
- Class imbalance significantly affects probe training
- Need for careful activation normalization
- Importance of proper architecture design

### Phase 2: Model Integration
- [x] NNSight integration for activation extraction
- [x] Memory-efficient activation storage
- [x] Layer selection strategy (every nth layer)
- [x] Basic probe training implementation
- [x] Enhanced probe architecture and training
- [ ] Comprehensive evaluation metrics

Step 43:
------
Implemented binary classification probes:
- Replaced multi-class classifier with binary classifiers
- One probe per moral foundation category
- Better handling of class imbalance
- Improved evaluation metrics

Key changes:
1. Probe Architecture:
   - Separate binary classifier for each moral foundation
   - Binary cross-entropy loss with class weighting
   - Sigmoid activation for probability output
   - Maintained neural architecture with hidden layer

2. Training Improvements:
   - Dynamic positive class weighting
   - Per-class F1 optimization
   - Independent training loops for each category
   - Better handling of class imbalance

3. Evaluation Enhancements:
   - Precision, recall, F1 for each category
   - Support counting for positive/negative samples
   - Threshold-based classification
   - Detailed per-class metrics

4. Storage Changes:
   - Separate model state for each category
   - Comprehensive metrics storage
   - Layer-wise organization of probes
   - Efficient probe management

### Current Status
- Implemented binary classification approach
  - One probe per moral foundation
  - Better handling of imbalanced classes
  - Independent optimization for each category
  - Improved evaluation metrics
- Enhanced training process
  - Class-specific weight adjustment
  - F1-score optimization
  - Detailed performance tracking
  - Threshold-based classification

### Next Steps
1. Evaluate binary classifier performance
   - Compare with multi-class approach
   - Analyze per-category performance
   - Study threshold sensitivity
   - Investigate failure cases

2. Optimize binary classifiers
   - Fine-tune thresholds
   - Experiment with architecture variants
   - Study learning dynamics
   - Analyze feature importance

3. Comprehensive analysis
   - Layer-wise performance patterns
   - Category-specific insights
   - Model behavior analysis
   - Correlation studies

4. Documentation and reporting
   - Performance comparisons
   - Insights documentation
   - Methodology description
   - Future recommendations

Step 44:
------
Simplified probe architecture to logistic regression:
- Replaced neural network with simple logistic classifier
- Fixed metrics calculation using sklearn
- Improved training stability and speed

Key changes:
1. Probe Architecture:
   - Single linear layer with sigmoid activation
   - Removed unnecessary complexity (normalization, dropout, hidden layers)
   - Direct logistic regression for binary classification
   - Faster training and more stable convergence

2. Training Improvements:
   - Increased batch size for faster training
   - Higher learning rate appropriate for logistic regression
   - Fewer epochs needed for convergence
   - Removed learning rate scheduling (unnecessary for logistic regression)

3. Metrics Calculation:
   - Using sklearn's metrics for reliable calculation
   - Fixed zero-division issues in metric computation
   - Added average probability monitoring
   - More stable and accurate metrics

4. Performance Optimization:
   - Faster training due to simpler architecture
   - More interpretable results
   - Better numerical stability
   - Reduced memory usage

### Current Status
- Implemented logistic regression probes
  - One probe per moral foundation
  - Simple and effective architecture
  - Stable training process
  - Reliable metrics calculation
- Enhanced efficiency
  - Faster training times
  - Lower memory usage
  - More interpretable results
  - Better numerical stability

### Next Steps
1. Evaluate logistic probe performance
   - Compare with previous approaches
   - Analyze per-category effectiveness
   - Study decision boundaries
   - Investigate feature importance

2. Analyze layer-wise patterns
   - Track performance across layers
   - Identify key representation layers
   - Study feature evolution
   - Map moral foundation encoding

3. Comprehensive analysis
   - Compare with baseline methods
   - Statistical significance testing
   - Error analysis
   - Feature importance study

4. Documentation and insights
   - Performance analysis
   - Layer-wise patterns
   - Feature importance
   - Recommendations for future work

Step 45:
------
Fixed tensor gradient handling in feature normalization:
- Added tensor detachment before numpy conversion
- Ensures proper handling of gradients in normalization pipeline
- Maintains numerical stability in feature scaling
- Prevents gradient computation errors

Key changes:
1. Tensor Operations:
   - Added detach() before numpy conversion
   - Proper handling of tensor gradients
   - Safe conversion between torch and numpy
   - Maintained data precision

2. Feature Normalization Pipeline:
   - Safe tensor detachment
   - Proper CPU transfer
   - Efficient numpy conversion
   - Memory-efficient processing

3. Implementation Details:
   - Updated _normalize_features method
   - Added gradient safety checks
   - Improved error handling
   - Better memory management

### Current Status
- Fixed gradient computation issues
  - Safe tensor operations
  - Proper gradient handling
  - Stable feature normalization
  - Memory-efficient processing
- Improved robustness
  - Better error handling
  - Safe data type conversions
  - Stable numerical operations
  - Efficient memory usage

### Next Steps
1. Monitor training stability
   - Track gradient flow
   - Verify feature scaling
   - Check memory usage
   - Analyze numerical stability

2. Performance optimization
   - Evaluate training speed
   - Monitor memory usage
   - Analyze computational efficiency
   - Fine-tune batch processing

3. Quality assurance
   - Verify normalization effects
   - Validate gradient handling
   - Test edge cases
   - Monitor system resources

Step 46:
------
Fixed loss function implementation:
- Switched to BCEWithLogitsLoss for proper class weighting
- Removed sigmoid from model (now part of loss)
- Improved numerical stability
- Better handling of class imbalance

Key changes:
1. Loss Function:
   - Using BCEWithLogitsLoss instead of BCELoss
   - Integrated sigmoid into loss computation
   - Better numerical stability
   - Proper class weight handling

2. Model Architecture:
   - Simplified LogisticProbe
   - Removed redundant sigmoid
   - Cleaner logits handling
   - More stable training

3. Training Process:
   - Improved gradient computation
   - Better numerical stability
   - Proper probability conversion
   - Consistent metric calculation

### Current Status
- Improved training stability
  - Better loss function
  - Proper class weighting
  - Stable gradient flow
  - Reliable metrics
- Enhanced architecture
  - Simplified model
  - Better numerical properties
  - Stable probability computation
  - Consistent evaluation

### Next Steps
1. Monitor training dynamics
   - Track loss convergence
   - Analyze class balance
   - Verify probability distributions
   - Check gradient behavior

2. Evaluate performance
   - Compare with previous results
   - Analyze class-wise metrics
   - Study probability calibration
   - Verify stability improvements

3. Fine-tune implementation
   - Optimize class weights
   - Adjust thresholds
   - Tune hyperparameters
   - Improve efficiency

Step 47:
------
Fixed loss function implementation:
- Switched to BCEWithLogitsLoss for proper class weighting
- Removed sigmoid from model (now part of loss)
- Improved numerical stability
- Better handling of class imbalance

Key changes:
1. Loss Function:
   - Using BCEWithLogitsLoss instead of BCELoss
   - Integrated sigmoid into loss computation
   - Better numerical stability
   - Proper class weight handling

2. Model Architecture:
   - Simplified LogisticProbe
   - Removed redundant sigmoid
   - Cleaner logits handling
   - More stable training

3. Training Process:
   - Improved gradient computation
   - Better numerical stability
   - Proper probability conversion
   - Consistent metric calculation

### Current Status
- Improved training stability
  - Better loss function
  - Proper class weighting
  - Stable gradient flow
  - Reliable metrics
- Enhanced architecture
  - Simplified model
  - Better numerical properties
  - Stable probability computation
  - Consistent evaluation

### Next Steps
1. Monitor training dynamics
   - Track loss convergence
   - Analyze class balance
   - Verify probability distributions
   - Check gradient behavior

2. Evaluate performance
   - Compare with previous results
   - Analyze class-wise metrics
   - Study probability calibration
   - Verify stability improvements

3. Fine-tune implementation
   - Optimize class weights
   - Adjust thresholds
   - Tune hyperparameters
   - Improve efficiency

Step 48:
------
Added comprehensive visualization functionality:
- Created visualization module for probe analysis
- Implemented layer-wise performance plots
- Added class distribution visualization
- Enhanced result interpretation

Key changes:
1. Visualization Module:
   - Layer-wise metric plots
   - Performance heatmaps
   - Class distribution plots
   - Weight distribution visualization

2. Metrics Visualization:
   - F1 score across layers
   - Precision and recall trends
   - Accuracy progression
   - Probability distribution

3. Distribution Analysis:
   - Class balance visualization
   - Positive/negative sample counts
   - Class weight distribution
   - Layer-wise patterns

4. Implementation Details:
   - Matplotlib/Seaborn integration
   - High-resolution output
   - Customizable plotting
   - Clear metric tracking

### Current Status
- Comprehensive visualization suite
  - Layer-wise performance tracking
  - Class distribution analysis
  - Metric progression plots
  - Intuitive visualizations
- Enhanced analysis capabilities
  - Visual performance tracking
  - Distribution insights
  - Pattern identification
  - Result interpretation

### Next Steps
1. Analyze visualization results
   - Identify performance patterns
   - Study layer-wise trends
   - Examine class imbalance effects
   - Track metric progression

2. Enhance visualizations
   - Add statistical annotations
   - Improve plot aesthetics
   - Include confidence intervals
   - Add interactive features

3. Documentation
   - Interpret visualization results
   - Document key findings
   - Update analysis insights
   - Prepare presentation materials

### Files Added
- `models/visualization.py`: Core visualization functionality
- `visualize_results.py`: Script to generate plots
- `figures/`: Directory for output visualizations

Step 49:
------
Enhanced visualization aesthetics and robustness:
- Removed seaborn dependency for better compatibility
- Improved plot aesthetics with native matplotlib
- Added value annotations to plots
- Enhanced readability and interpretation

Key changes:
1. Plot Aesthetics:
   - Custom color schemes
   - Improved font sizes and spacing
   - Better grid styling
   - Value annotations on plots

2. Visualization Features:
   - Direct value labels on bars
   - Customized heatmap coloring
   - Improved legend placement
   - Better axis formatting

3. Technical Improvements:
   - Removed external style dependencies
   - More robust plotting functions
   - Better color contrast
   - Enhanced readability

4. User Experience:
   - Clearer data presentation
   - More intuitive visualizations
   - Better value interpretation
   - Higher quality output

### Current Status
- Enhanced visualization system
  - Better plot aesthetics
  - More informative displays
  - Robust implementation
  - High-quality output
- Improved usability
  - Clear value presentation
  - Intuitive color schemes
  - Better readability
  - Consistent styling

### Next Steps
1. Analyze visualization output
   - Review plot clarity
   - Check value readability
   - Verify color effectiveness
   - Test with different data scales

2. Further enhancements
   - Add statistical annotations
   - Consider interactive features
   - Explore additional plot types
   - Improve accessibility

3. Documentation
   - Update usage guidelines
   - Document style choices
   - Add interpretation guide
   - Include example outputs

### Dependencies
Required Python packages:
- numpy
- matplotlib
- torch
- (removed seaborn dependency)

Step 50:
------
Corrected probe training approach:
- Removed incorrect class weighting
- Fixed sample selection for binary classification
- Improved training focus for each moral foundation
- Enhanced data efficiency

Key changes:
1. Sample Selection:
   - Only use relevant samples for each class
   - No artificial negative samples
   - Better focus on actual class characteristics
   - More efficient training

2. Training Process:
   - Removed unnecessary class weights
   - Simplified loss function
   - More focused learning
   - Better data utilization

3. Implementation Details:
   - Updated sample filtering
   - Removed weight calculation
   - Improved sample tracking
   - Better logging of sample counts

4. Conceptual Improvements:
   - Better alignment with task goals
   - More accurate representation learning
   - Cleaner training signal
   - Improved model focus

### Current Status
- Corrected training approach
  - Proper sample selection
  - No artificial imbalance
  - Focused learning per class
  - Better data efficiency
- Improved methodology
  - Cleaner implementation
  - Better theoretical foundation
  - More accurate evaluation
  - Efficient resource use

### Next Steps
1. Evaluate new approach
   - Compare performance
   - Analyze training dynamics
   - Check convergence
   - Verify improvements

2. Further optimization
   - Fine-tune hyperparameters
   - Study sample efficiency
   - Analyze model behavior
   - Improve training speed

3. Documentation
   - Update methodology description
   - Document improvements
   - Explain rationale
   - Share findings

### Impact
This change should lead to:
1. More accurate probes
2. Better training efficiency
3. Cleaner learning signals
4. More interpretable results

Step 51:
------
Implemented balanced sampling for binary classification:
- Added balanced negative sampling
- Equal positive and negative samples per class
- Random selection of negative examples
- Improved training signal

Key changes:
1. Sample Selection:
   - Equal number of positive and negative samples
   - Random selection of negative examples
   - Balanced training sets
   - Better class representation

2. Training Process:
   - No need for class weights (naturally balanced)
   - Better discrimination learning
   - More robust decision boundaries
   - Improved generalization

3. Implementation Details:
   - Random negative sampling
   - Balanced validation sets
   - Better sample tracking
   - Improved logging

4. Conceptual Improvements:
   - True binary classification
   - Better foundation discrimination
   - More reliable evaluation
   - Proper negative examples

### Current Status
- Balanced training approach
  - Equal class representation
  - Random negative sampling
  - Proper binary classification
  - Better evaluation metrics
- Improved methodology
  - Better theoretical foundation
  - More reliable training
  - Proper discrimination learning
  - Efficient resource use

### Next Steps
1. Evaluate balanced approach
   - Compare with previous results
   - Analyze discrimination ability
   - Check negative sample impact
   - Verify training stability

2. Fine-tune implementation
   - Study negative sampling strategies
   - Optimize batch composition
   - Analyze training dynamics
   - Improve efficiency

3. Analysis and documentation
   - Compare approaches
   - Document improvements
   - Study negative examples
   - Share insights

### Impact
This change should lead to:
1. Better discrimination between foundations
2. More reliable probe performance
3. More meaningful evaluation metrics
4. Better generalization
