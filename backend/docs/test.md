# Test Documentation

This document provides a comprehensive overview of all tests in the `tests/` directory, explaining their purpose, coverage, and implementation details.

---

## Table of Contents

1. [Test Decision Aggregation](#test-decision-aggregation)
2. [Test Decision Calibration](#test-decision-calibration)
3. [Test Decision Temporal](#test-decision-temporal)
4. [Test Decision Integration](#test-decision-integration)
5. [Test Inference](#test-inference)
6. [Test Models](#test-models)
7. [Test Preprocessing](#test-preprocessing)

---

## Test Decision Aggregation

**File:** `tests/test_decision_aggregation.py`

### Purpose
Comprehensive tests for the Structural Aggregation Engine, which combines multiple urgency metrics (mean, max, and percentile) to compute an overall structural urgency score for complaint clusters.

### Test Suites

#### 1. TestMeanComputation
Tests the correctness of mean urgency computation.

**Tests:**
- `test_mean_computation_simple`: Verifies mean calculation with simple values (0.2, 0.4, 0.6)
- `test_mean_computation_all_same`: Tests mean when all urgency values are identical (0.7)
- `test_mean_computation_extremes`: Tests mean with extreme values (0.0 and 1.0)

**Key Validation:** Ensures mean urgency is calculated correctly using standard averaging formula.

#### 2. TestMaxDetection
Tests the detection of maximum urgency values.

**Tests:**
- `test_max_detection_correctness`: Verifies max is correctly identified in middle of dataset
- `test_max_at_beginning`: Tests max detection when it's the first element
- `test_max_at_end`: Tests max detection when it's the last element

**Key Validation:** Confirms that maximum urgency is accurately identified regardless of position.

#### 3. TestPercentileComputation
Tests the 90th percentile calculation.

**Tests:**
- `test_percentile_90_standard`: Tests 90th percentile with 10 complaints (urgencies 0.1 to 1.0)
- `test_percentile_small_dataset`: Tests percentile with only 2 complaints

**Key Validation:** Verifies that percentile calculation handles both large and small datasets correctly.

#### 4. TestSingleComplaint
Tests edge case of single-complaint clusters.

**Tests:**
- `test_single_complaint_cluster`: Verifies that mean, max, and percentile all equal the single value (0.75)
- `test_single_complaint_structural_urgency`: Tests structural urgency with equal weights

**Key Validation:** Ensures single-complaint clusters don't cause errors and produce sensible results.

#### 5. TestIdenticalUrgency
Tests clusters where all complaints have identical urgency scores.

**Tests:**
- `test_all_identical_urgency`: Tests 10 complaints all with urgency 0.5
- `test_identical_high_urgency`: Tests with all urgency at 0.95
- `test_identical_low_urgency`: Tests with all urgency at 0.1

**Key Validation:** Confirms that when all values are identical, all statistics equal that value.

#### 6. TestExtremeOutliers
Tests handling of extreme outlier cases.

**Tests:**
- `test_single_extreme_outlier`: Tests cluster with one 1.0 urgency among four 0.1 values
- `test_multiple_outliers`: Tests cluster with multiple high outliers

**Key Validation:** Verifies that outliers are captured in max and percentile but don't unfairly dominate the structural urgency.

#### 7. TestWeightConfiguration
Tests different weight configurations for aggregation.

**Tests:**
- `test_max_dominant_weights`: Tests config with β=0.8 (max-dominated)
- `test_mean_dominant_weights`: Tests config with α=0.8 (mean-dominated)

**Key Validation:** Ensures weight configuration properly influences the final structural urgency.

#### 8. TestErrorHandling
Tests error handling and validation.

**Tests:**
- `test_empty_cluster_raises_error`: Verifies ValueError on empty complaint list
- `test_invalid_urgency_score_raises_error`: Tests detection of urgency > 1.0
- `test_negative_urgency_score_raises_error`: Tests detection of negative urgency

**Key Validation:** Ensures robust error handling for invalid inputs.

#### 9. TestAggregateMultiple
Tests batch aggregation of multiple clusters.

**Tests:**
- `test_aggregate_multiple_clusters`: Tests aggregating two clusters (CLUSTER_A and CLUSTER_B) simultaneously

**Key Validation:** Confirms that multiple clusters can be processed in batch correctly.

#### 10. TestConvenienceFunction
Tests the convenience wrapper function.

**Tests:**
- `test_compute_structural_urgency_function`: Tests the `compute_structural_urgency()` function

**Key Validation:** Verifies the convenience function works and returns valid float in [0, 1].

---

## Test Decision Calibration

**File:** `tests/test_decision_calibration.py`

### Purpose
Tests the Urgency Calibration Engine, which applies size-based penalties and temporal smoothing to prevent large clusters from unfairly dominating and to reduce volatility.

### Test Suites

#### 1. TestSizeNormalization
Tests size-based penalty mechanism.

**Tests:**
- `test_larger_cluster_gets_dampened`: Compares small cluster (5 complaints) vs large cluster (100 complaints) with same raw urgency (0.8)
- `test_single_complaint_no_penalty`: Tests that single-complaint clusters receive minimal penalty

**Key Validation:** Ensures larger clusters receive stronger dampening penalty to prevent them from unfairly dominating smaller, potentially more critical clusters.

#### 2. TestSmoothing
Tests temporal smoothing to reduce volatility.

**Tests:**
- `test_smoothing_reduces_volatility`: Tests two consecutive calibrations (0.5 → 0.9) separated by 1 hour

**Key Validation:** Verifies that smoothing prevents sudden jumps in urgency scores by incorporating previous urgency values.

#### 3. TestInvalidInput
Tests error handling for invalid inputs.

**Tests:**
- `test_invalid_urgency_value`: Tests that urgency > 1.0 raises ValueError
- `test_zero_urgency`: Tests that zero urgency is handled correctly

**Key Validation:** Ensures robust input validation and appropriate error handling.

---

## Test Decision Temporal

**File:** `tests/test_decision_temporal.py`

### Purpose
Tests the Temporal Urgency Engine, which analyzes complaint arrival patterns to detect bursts and compute temporal urgency based on complaint frequency dynamics.

### Helper Functions
- `create_complaint()`: Creates test complaints with specified timestamps (hours_ago parameter)

### Test Suites

#### 1. TestBurstDetection
Tests burst detection and its impact on temporal urgency.

**Tests:**
- `test_recent_burst_increases_score`: Tests scenario with 2 historical complaints vs 10 recent complaints (5x increase)
- `test_no_burst_low_score`: Tests evenly distributed complaints (no burst pattern)

**Key Validation:** 
- Burst scenarios should have volume_ratio > 1.0 and temporal_urgency > 0.5
- Steady arrival should have volume_ratio < 2.0 and temporal_urgency < 0.8

#### 2. TestSingleComplaint
Tests edge case of single complaint.

**Tests:**
- `test_single_complaint_handled`: Tests that single complaint doesn't cause errors

**Key Validation:** 
- Arrival rate and mean inter-arrival should be 0.0
- Should return valid urgency in [0, 1] without crashing

#### 3. TestErrorHandling
Tests error handling for invalid inputs.

**Tests:**
- `test_empty_complaints_raises_error`: Verifies ValueError on empty complaint list

**Key Validation:** Ensures proper error handling for invalid inputs.

---

## Test Decision Integration

**File:** `tests/test_decision_integration.py`

### Purpose
End-to-end integration tests for the complete Decision Engine pipeline, validating that the orchestrator correctly combines structural urgency, temporal urgency, and calibration.

### Helper Functions
- `create_complaint()`: Creates test complaints with configurable cluster_id, urgency, and timestamp

### Test Suites

#### 1. TestValidationGoal1
Validates that increasing complaint severity increases structural urgency.

**Tests:**
- `test_higher_severity_higher_urgency`: Compares LOW_SEV cluster (urgency 0.3) vs HIGH_SEV cluster (urgency 0.8)

**Key Validation:** 
- High severity cluster should have higher structural urgency
- High severity cluster should rank first in sorted results

#### 2. TestFullPipeline
Tests the complete pipeline with realistic scenarios.

**Tests:**
- `test_full_pipeline`: Tests 3 clusters with different characteristics:
  - CLUSTER_A: High severity (0.9), low volume (3 complaints)
  - CLUSTER_B: Medium severity (0.6), high burst (15 complaints in short time)
  - CLUSTER_C: Low severity (0.3), medium volume (7 complaints)

**Key Validation:**
- Should process all 3 clusters successfully
- All final urgency scores should be in valid range [0, 1]
- Ranking should reflect combined structural and temporal factors

#### 3. TestErrorHandling
Tests error handling in the orchestrator.

**Tests:**
- `test_empty_complaints`: Verifies ValueError on empty complaint list

**Key Validation:** Ensures proper error handling at the orchestrator level.

---

## Test Inference

**File:** `tests/test_inference.py`

### Purpose
Tests the Urgency Decision Engine, which makes final urgency decisions by applying rule-based adjustments on top of model predictions.

### Test Suites

#### 1. TestDecisionEngine
Tests the core decision-making functionality.

**Tests:**
- `test_basic_decision`: Tests basic decision with Medium urgency prediction (confidence 0.6)
- `test_temporal_boost`: Tests nighttime temporal boost (is_night=True)
- `test_keyword_detection`: Tests detection of critical keywords ("emergency", "broken pipe", "flooding")
- `test_history_boost`: Tests boost for repeated complaints (2 previous water-related complaints)

**Key Validation:**
- Basic decisions should include urgency_level, confidence, and adjustments
- Temporal features should boost confidence
- Critical keywords should trigger keyword-type adjustments
- Repeated complaints should boost confidence

**Decision Adjustment Types:**
- `temporal`: Boosts for night/peak hours/weekends
- `keyword`: Boosts for emergency keywords
- `history`: Boosts for repeated complaints

---

## Test Models

**File:** `tests/test_models.py`

### Purpose
Tests neural network model architectures to ensure they can be created, compiled, and produce correct output shapes.

### Test Configuration
All tests use consistent parameters:
- `vocab_size`: 1000
- `embedding_dim`: 100
- `max_length`: 50
- `num_classes`: 3

### Test Suites

#### 1. TestModels
Tests all model architectures.

**Tests:**
- `test_cnn_model_creation`: Tests CNN model creation and output shape
- `test_bilstm_model_creation`: Tests BiLSTM model creation and output shape
- `test_cnn_bilstm_model_creation`: Tests combined CNN-BiLSTM model creation and output shape
- `test_dense_head_creation`: Tests dense head model with dual inputs (text features + temporal features)

**Key Validation:**
- Models can be created without errors
- Models can be compiled
- Output shapes are correct:
  - CNN, BiLSTM, CNN-BiLSTM: (1, num_classes)
  - DenseHead: (1, num_classes) with dual inputs

**Architectures Tested:**
1. **CNNModel**: Convolutional neural network for text classification
2. **BiLSTMModel**: Bidirectional LSTM for sequence processing
3. **CNNBiLSTMModel**: Hybrid model combining CNN and BiLSTM
4. **DenseHead**: Fully connected network combining text and temporal features

---

## Test Preprocessing

**File:** `tests/test_preprocessing.py`

### Purpose
Tests text preprocessing components including text cleaning and tokenization with urgency-aware features.

### Test Suites

#### 1. TestTextCleaner
Tests the TextCleaner class for text normalization and feature extraction.

**Tests:**
- `test_basic_cleaning`: Tests basic text cleaning functionality
- `test_caps_detection`: Tests detection of ALL-CAPS text (adds "[caps_on]" marker)
- `test_emoji_conversion`: Tests emoji-to-text conversion (😊 → text)
- `test_punctuation_barrage`: Tests excessive punctuation detection (e.g., "!!!" → "multiple exclamations")
- `test_empty_text`: Tests handling of empty strings
- `test_contraction_expansion`: Tests contraction expansion ("can't" → "cannot")

**Key Validation:**
- Text cleaning preserves urgency signals
- Special markers are added for urgency indicators (caps, punctuation)
- Emojis are converted to descriptive text
- Contractions are properly expanded

#### 2. TestTokenizer
Tests the Tokenise class for urgency-aware tokenization.

**Tests:**
- `test_basic_tokenization`: Tests basic tokenization returns list of tokens
- `test_urgency_word_preservation`: Tests that urgency-critical words like "not" and "now" are preserved (not removed as stop words)
- `test_special_tag_preservation`: Tests that special tags like "[Multiple Exclamations]" are preserved as single tokens

**Key Validation:**
- Tokenization produces list output
- Urgency-critical words are NOT filtered out as stop words
- Special urgency markers are preserved as single tokens

**Urgency Features:**
- Caps detection: Indicates shouting/emphasis
- Punctuation barrage: Indicates urgency/emotion
- Urgency word preservation: Keeps negations and temporal indicators
- Special tag preservation: Maintains preprocessing markers

---

## Running the Tests

### Run All Tests
```bash
cd tests
pytest -v
```

### Run Specific Test File
```bash
pytest tests/test_decision_aggregation.py -v
pytest tests/test_decision_calibration.py -v
pytest tests/test_decision_temporal.py -v
pytest tests/test_decision_integration.py -v
pytest tests/test_inference.py -v
pytest tests/test_models.py -v
pytest tests/test_preprocessing.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_decision_aggregation.py::TestMeanComputation -v
```

### Run Specific Test Method
```bash
pytest tests/test_decision_aggregation.py::TestMeanComputation::test_mean_computation_simple -v
```

---

## Test Coverage Summary

| Component | Test File | Test Classes | Key Areas Covered |
|-----------|-----------|--------------|-------------------|
| **Structural Aggregation** | test_decision_aggregation.py | 10 | Mean/max/percentile computation, outliers, weights, error handling |
| **Urgency Calibration** | test_decision_calibration.py | 3 | Size normalization, temporal smoothing, input validation |
| **Temporal Analysis** | test_decision_temporal.py | 3 | Burst detection, single complaint edge case, error handling |
| **Integration** | test_decision_integration.py | 3 | End-to-end pipeline, severity ranking, full system validation |
| **Inference Engine** | test_inference.py | 1 | Decision making, temporal boost, keyword detection, history |
| **Model Architectures** | test_models.py | 1 | CNN, BiLSTM, CNN-BiLSTM, DenseHead creation and outputs |
| **Preprocessing** | test_preprocessing.py | 2 | Text cleaning, tokenization, urgency feature preservation |

---

## Test Principles

### 1. Comprehensive Coverage
- **Edge Cases**: Single complaint, empty clusters, identical values
- **Extremes**: Outliers, maximum/minimum values, large datasets
- **Errors**: Invalid inputs, boundary violations, type errors

### 2. Validation Goals
- **Correctness**: Mathematical accuracy of calculations
- **Robustness**: Graceful handling of edge cases
- **Stability**: Consistent behavior across scenarios
- **Integration**: End-to-end pipeline validation

### 3. Test Categories
- **Unit Tests**: Individual components (aggregation, calibration, temporal)
- **Integration Tests**: Complete pipeline (orchestrator)
- **Model Tests**: Architecture validation
- **Preprocessing Tests**: Text processing pipeline

### 4. Assertion Strategies
- **Exact Matching**: For deterministic calculations (using `assertAlmostEqual` with places=6)
- **Range Validation**: For probabilistic outputs (checking [0, 1] bounds)
- **Comparative Checks**: For ranking and ordering
- **Error Detection**: Using `assertRaises` for invalid inputs

---

## Dependencies

The test suite requires:
- `unittest`: Standard Python testing framework
- `pytest`: Used for some tests (fixtures, markers)
- `numpy`: Numerical computations
- `datetime`: Temporal features

---

## Future Test Enhancements

### Potential Additions
1. **Performance Tests**: Measure execution time for large complaint volumes
2. **Stress Tests**: Test with extreme cluster sizes (1000+ complaints)
3. **Parameterized Tests**: Use pytest.mark.parametrize for broader coverage
4. **Mock Tests**: Mock data interface for isolated component testing
5. **Regression Tests**: Capture and validate against known outputs
6. **Coverage Reports**: Add pytest-cov for coverage analysis

### Test Maintenance
- Keep tests synchronized with production code changes
- Update test data when schema changes
- Maintain test documentation alongside code changes
- Review and update edge cases as new scenarios emerge

---

**Last Updated:** February 21, 2026  
**Total Test Files:** 7  
**Estimated Total Tests:** 50+
