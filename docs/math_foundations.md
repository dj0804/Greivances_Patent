# Mathematical Foundations of the Decision Engine

## Overview

This document provides the complete mathematical foundation for urgency computation, including formulas, justifications, bias correction reasoning, and stability guarantees.

## Table of Contents

1. [Notation and Definitions](#notation-and-definitions)
2. [Structural Urgency Aggregation](#structural-urgency-aggregation)
3. [Temporal Urgency Analysis](#temporal-urgency-analysis)
4. [Raw Urgency Fusion](#raw-urgency-fusion)
5. [Calibration and Bias Correction](#calibration-and-bias-correction)
6. [Numerical Stability](#numerical-stability)
7. [Theoretical Properties](#theoretical-properties)

---

## Notation and Definitions

### Cluster k

Let cluster $k$ contain $n_k$ complaints: $\{c_1, c_2, \ldots, c_{n_k}\}$

For each complaint $c_i$:
- $U_i \in [0, 1]$: Urgency score
- $t_i$: Timestamp
- $\text{text}_i$: Complaint text

### Time Windows

- $T_{\text{recent}}$: Recent time window (default: 24 hours)
- $T_{\text{historical}}$: Historical time window (default: 168 hours = 1 week)
- $t_{\text{ref}}$: Reference time (typically "now")

---

## Structural Urgency Aggregation

### Formula

$$U_{\text{struct},k} = \alpha \cdot \mu_k + \beta \cdot M_k + \delta \cdot P_k$$

where:

- $\mu_k = \frac{1}{n_k} \sum_{i=1}^{n_k} U_i$ (mean urgency)
- $M_k = \max(U_1, U_2, \ldots, U_{n_k})$ (maximum urgency)
- $P_k = \text{percentile}_{90}(U_1, \ldots, U_{n_k})$ (90th percentile)

with constraint: $\alpha + \beta + \delta = 1$

**Default values**: $\alpha = 0.4, \beta = 0.3, \delta = 0.3$

### Justification

#### Why Three Components?

1. **Mean ($\mu_k$)**: Represents overall cluster severity
   - Reflects average urgency across all complaints
   - Captures general sentiment of the cluster
   
2. **Maximum ($M_k$)**: Captures worst-case scenario
   - Ensures critical outliers aren't buried in averaging
   - Important for identifying clusters with at least one critical issue
   
3. **90th Percentile ($P_k$)**: Robust high-severity measure
   - Less sensitive to single extreme outliers than max
   - Indicates prevalence of high-urgency complaints
   - More stable than max for ranking

#### Why This Weighted Combination?

The weighted combination balances three perspectives:

- **Robustness**: Percentile is robust to outliers
- **Sensitivity**: Max captures critical cases
- **Representativeness**: Mean represents typical urgency

Alternative considered: Median instead of mean
- **Rejected** because median can mask bimodal distributions

### Properties

**Theorem 1**: $U_{\text{struct},k} \in [0, 1]$ if all $U_i \in [0, 1]$

*Proof*: Since $\mu_k, M_k, P_k \in [0, 1]$ and $\alpha + \beta + \delta = 1$ with non-negative weights, the convex combination preserves the range. ∎

**Theorem 2**: $U_{\text{struct},k}$ is monotonically non-decreasing in each $U_i$

*Proof*: Increasing any $U_i$ can only increase (or maintain) $\mu_k$, $M_k$, and $P_k$, thus increasing the weighted sum. ∎

---

## Temporal Urgency Analysis

### Component 1: Volume Ratio

$$R_k = \frac{V_{\text{recent}}}{V_{\text{historical}} + \epsilon}$$

where:
- $V_{\text{recent}} = |\{c_i : t_i \geq t_{\text{ref}} - T_{\text{recent}}\}|$
- $V_{\text{historical}} = |\{c_i : t_{\text{ref}} - T_{\text{historical}} \leq t_i < t_{\text{ref}} - T_{\text{recent}}\}|$
- $\epsilon = 10^{-6}$ (smoothing constant)

**Interpretation**:
- $R_k > 1$: Recent surge (burst detection)
- $R_k = 1$: Steady arrival
- $R_k < 1$: Declining volume

### Component 2: Arrival Rate

For $n_k > 1$:

$$\lambda_k = \frac{n_k - 1}{t_{\max} - t_{\min}} \times 3600$$

measured in complaints per hour.

For $n_k = 1$: $\lambda_k = 0$

**Interpretation**:
- High $\lambda_k$: Rapid complaint submission
- Low $\lambda_k$: Slow trickle

### Component 3: Temporal Intensity

$$T_k = \sigma(\theta_1 \cdot R_k + \theta_2 \cdot \lambda_k)$$

where $\sigma(x)$ is the sigmoid function:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Default values**: $\theta_1 = 2.0, \theta_2 = 1.5$

### Justification

#### Why Sigmoid?

The sigmoid function provides:

1. **Smoothness**: Continuously differentiable
2. **Boundedness**: Maps $\mathbb{R} \to (0, 1)$
3. **Interpretability**: S-curve mimics "urgency perception"
4. **Saturation**: Prevents extreme values from dominating

#### Why These Features?

- **Volume Ratio**: Detects sudden changes (bursts)
- **Arrival Rate**: Measures absolute frequency
- **Linear Combination**: Allows independent tuning of burst vs frequency sensitivity

### Properties

**Theorem 3**: $T_k \in (0, 1)$ for all finite inputs

*Proof*: Sigmoid maps $\mathbb{R} \to (0, 1)$, and the linear combination is finite for finite $R_k, \lambda_k$. ∎

**Theorem 4**: $T_k$ is monotonically increasing in both $R_k$ and $\lambda_k$

*Proof*: Sigmoid is monotonically increasing, and both coefficients $\theta_1, \theta_2 > 0$. ∎

### Numerical Stability

The sigmoid can overflow for large $|x|$. We use:

$$\sigma(x) = \begin{cases}
\frac{1}{1 + e^{-x}} & \text{if } x \geq 0 \\
\frac{e^x}{1 + e^x} & \text{if } x < 0
\end{cases}$$

This prevents overflow: for $x \to -\infty$, we compute $e^x$ (safe) rather than $e^{-x}$ (overflow).

---

## Raw Urgency Fusion

### Formula

$$U_{\text{raw},k} = \lambda_1 \cdot U_{\text{struct},k} + \lambda_2 \cdot T_k$$

with constraint: $\lambda_1 + \lambda_2 = 1$

**Default values**: $\lambda_1 = 0.6, \lambda_2 = 0.4$

### Justification

This is a weighted fusion of:
- **Structural urgency**: What is being complained about (severity)
- **Temporal urgency**: When and how fast (dynamics)

The 60-40 split gives more weight to content (what) than timing (when), based on the principle that severity matters more than speed for prioritization.

### Properties

**Theorem 5**: $U_{\text{raw},k} \in [0, 1]$ if inputs are valid

*Proof*: Convex combination of values in $[0, 1]$ remains in $[0, 1]$. ∎

---

## Calibration and Bias Correction

### Problem Statement

**Bias**: Large clusters naturally have higher max and mean values simply due to sample size, not necessarily because of higher true urgency.

**Solution**: Apply logarithmic penalty inversely proportional to cluster size.

### Size Normalization

$$U_{\text{size},k} = \frac{U_{\text{raw},k}}{1 + \log(1 + n_k)}$$

### Justification for Logarithmic Penalty

#### Why Logarithm?

1. **Diminishing Returns**: Each additional complaint contributes less penalty
   - $n = 10$: penalty $\approx 3.4$
   - $n = 100$: penalty $\approx 5.6$ (not 34)
   - $n = 1000$: penalty $\approx 7.9$ (not 340)

2. **Single Complaint**: No penalty for $n = 1$
   - $\log(1 + 1) = \log(2) \approx 0.69$
   - Penalty factor: $1 + 0.69 = 1.69$ (modest)

3. **Fairness**: Allows small high-urgency clusters to outrank large moderate ones

#### Alternative Considered: Linear Penalty

$$U_{\text{size}} = \frac{U_{\text{raw}}}{1 + \alpha \cdot n_k}$$

**Rejected** because:
- Too harsh for large clusters
- $n = 100$ would get penalty factor 101 (excessive)

#### Alternative Considered: Square Root

$$U_{\text{size}} = \frac{U_{\text{raw}}}{1 + \sqrt{n_k}}$$

**Rejected** because:
- Not aggressive enough for very large clusters
- Still allows size to dominate

### Temporal Smoothing

$$U_{\text{final},k} = \gamma \cdot U_{\text{prev},k} + (1 - \gamma) \cdot U_{\text{size},k}$$

where $\gamma \in [0, 1]$ is the smoothing factor.

**Default**: $\gamma = 0.7$

### Justification for Exponential Smoothing

Without smoothing, urgency can oscillate wildly between updates:
- $t_1$: Urgency = 0.5
- $t_2$: Urgency = 0.9 (sudden jump)
- $t_3$: Urgency = 0.3 (sudden drop)

This creates unstable rankings and user confusion.

Exponential smoothing provides:

1. **Continuity**: Gradual transitions
2. **Stability**: Reduces variance
3. **Responsiveness**: Still adapts to real changes (with $\gamma < 1$)

#### Choice of $\gamma = 0.7$

- $\gamma = 0$: No memory (too reactive)
- $\gamma = 0.5$: Equal weight to past and present
- $\gamma = 0.7$: Strong but not rigid memory
- $\gamma = 0.9$: Too sluggish (slow to adapt)
- $\gamma = 1$: No updates (frozen)

**Rationale**: 0.7 balances stability (70% weight on past) with adaptability (30% weight on new).

### Properties

**Theorem 6**: Size normalization reduces urgency for $n_k > 1$

*Proof*: For $n_k > 1$, $1 + \log(1 + n_k) > 1$, so division reduces the value. ∎

**Theorem 7**: Smoothing reduces variance

Let $\sigma^2_{\text{raw}}$ be variance of raw urgencies over time, and $\sigma^2_{\text{smooth}}$ be variance after smoothing.

*Claim*: $\sigma^2_{\text{smooth}} < \sigma^2_{\text{raw}}$ for $0 < \gamma < 1$

*Proof sketch*: Exponential smoothing is a low-pass filter; it attenuates high-frequency components (oscillations). Variance is reduced proportionally to $(1-\gamma)$ factor. ∎

---

## Numerical Stability

### Division by Zero Prevention

1. **Volume Ratio**: $V_{\text{historical}} + \epsilon$ ensures denominator $\geq \epsilon > 0$

2. **Arrival Rate**: Special case for $n_k = 1$ returns 0, avoiding $0/0$

3. **Sigmoid**: Clipping to $[-30, 30]$ prevents $e^{x}$ overflow

### Floating Point Considerations

All computations use IEEE 754 double precision:
- Range: $\pm 10^{308}$
- Precision: $\sim 15$ decimal digits

Operations guaranteed not to overflow:
- Convex combinations (weighted averages)
- Logarithms of positive values
- Sigmoid with clipping

### Verification

We verify:
- All outputs $\in [0, 1]$ (clamped if necessary)
- No NaN or Inf values
- Deterministic output for same input

---

## Theoretical Properties

### Monotonicity

**Property**: Increasing individual complaint urgency $U_i$ increases (or maintains) final cluster urgency $U_{\text{final},k}$

*Proof*: Follows from monotonicity of aggregation, fusion, and calibration steps.

### Boundedness

**Property**: $U_{\text{final},k} \in [0, 1]$ for all valid inputs

*Guaranteed by*:
- Input validation
- Convex combinations
- Clamping in calibration

### Stability

**Property**: Small changes in input produce small changes in output

*Achieved through*:
- Continuous functions (no discontinuities)
- Sigmoid smoothness
- Exponential smoothing

### Fairness

**Property**: Cluster size alone does not determine ranking

*Ensured by*: Logarithmic size penalty preventing large clusters from dominating

---

## Example Calculations

### Example 1: Small High-Urgency Cluster

**Inputs**:
- $n = 5$
- Urgencies: $[0.9, 0.85, 0.9, 0.88, 0.92]$
- Temporal: $T = 0.6$

**Structural**:
- $\mu = 0.89$
- $M = 0.92$
- $P_{90} = 0.91$
- $U_{\text{struct}} = 0.4(0.89) + 0.3(0.92) + 0.3(0.91) = 0.905$

**Raw Fusion**:
- $U_{\text{raw}} = 0.6(0.905) + 0.4(0.6) = 0.783$

**Calibration**:
- Penalty: $1 + \log(6) \approx 2.79$
- $U_{\text{size}} = 0.783 / 2.79 \approx 0.281$

**Final** (no previous): $U_{\text{final}} = 0.281$

### Example 2: Large Moderate-Urgency Cluster

**Inputs**:
- $n = 100$
- Urgencies: mean $= 0.7$, max $= 0.9$, $P_{90} = 0.85$
- Temporal: $T = 0.7$

**Structural**:
- $U_{\text{struct}} = 0.4(0.7) + 0.3(0.9) + 0.3(0.85) = 0.805$

**Raw Fusion**:
- $U_{\text{raw}} = 0.6(0.805) + 0.4(0.7) = 0.763$

**Calibration**:
- Penalty: $1 + \log(101) \approx 5.62$
- $U_{\text{size}} = 0.763 / 5.62 \approx 0.136$

**Final**: $U_{\text{final}} = 0.136$

**Conclusion**: Despite higher raw urgency, the large cluster ranks lower than the small cluster due to size penalty. This is intentional - the penalty ensures that true severity per complaint, not volume, drives ranking.

---

## Sensitivity Analysis

### Varying Aggregation Weights

| $\alpha$ | $\beta$ | $\delta$ | Behavior |
|----------|---------|----------|----------|
| 0.8 | 0.1 | 0.1 | Mean-dominated (outlier-resistant) |
| 0.1 | 0.8 | 0.1 | Max-dominated (sensitive to peaks) |
| 0.33 | 0.33 | 0.34 | Balanced |

### Varying Fusion Weights

| $\lambda_1$ | $\lambda_2$ | Behavior |
|-------------|-------------|----------|
| 0.9 | 0.1 | Content-focused (severity over timing) |
| 0.5 | 0.5 | Balanced |
| 0.3 | 0.7 | Time-focused (bursts over severity) |

---

## Conclusion

The mathematical framework provides:

1. **Sound theoretical foundation**: All formulas mathematically justified
2. **Bias correction**: Logarithmic penalty prevents size dominance  
3. **Stability**: Smoothing reduces volatility
4. **Robustness**: Numerical stability guaranteed
5. **Configurability**: Weights allow tuning behavior

This ensures the Decision Engine produces reliable, interpretable, and fair urgency rankings.
