# DataDNA 🧬

> Your dataset has a unique genetic signature. DataDNA reveals it.

DataDNA is a synthetic data quality evaluation framework that creates unique **DNA fingerprint signatures** for datasets. It measures how well synthetic data preserves the statistical "genetics" of real data across multiple dimensions: distributions, correlations, outliers, entropy, and ML utility.

## What It Does

- Generates a **DNA Fingerprint** (unique hash-based signature) for any dataset
- Compares **real vs synthetic** data fidelity across 12+ statistical dimensions
- Detects **mode collapse** in GAN-generated data
- Measures **correlation preservation** using novel entropic distance metrics
- Scores **outlier fidelity** — does synthetic data preserve rare patterns?
- Evaluates **ML utility** — can a model trained on synthetic data perform on real data?
- Produces a **Data DNA Report** with interactive visualizations

## Why This Project Is Unique

| Existing Tools | DataDNA |
|---|---|
| `SDV` / `CTGAN` evaluate synthetic data | **DNA fingerprint** metaphor with visual encoding |
| Basic statistical tests (KS, chi-squared) | **12-dimensional quality score** with weighted composite |
| Single-metric evaluation | **ML utility scoring** + mode collapse detection |
| None produce visual "DNA" signatures | Unique **visual DNA helix** encoding per dataset |

## Installation

```bash
pip install datadna
```

## Quick Start

```python
import pandas as pd
from datadna import DataDNA, DNALoader

# Load your data
real_data = pd.read_csv("real_data.csv")
synthetic_data = pd.read_csv("synthetic_data.csv")  # from GAN/VAE/etc.

# Create DNA fingerprints
real_fingerprint = DataDNA(real_data)
synthetic_fingerprint = DataDNA(synthetic_data)

# Compare
comparison = real_fingerprint.compare(synthetic_fingerprint)

print(f"Overall Quality Score: {comparison.quality_score}/100")
print(f"Fidelity Grade: {comparison.grade}")  # A+, A, B+, etc.
print(f"Mode Collapse Detected: {comparison.mode_collapse}")
print(f"ML Utility Score: {comparison.ml_utility_score}/100")

# Generate visual report
comparison.generate_report("data_dna_report.html")

# Get the DNA string representation
print(f"Real DNA: {real_fingerprint.dna_string}")
print(f"Synthetic DNA: {synthetic_fingerprint.dna_string}")
```

### CLI Usage

```bash
# Compare real vs synthetic
datadna compare real_data.csv synthetic_data.csv --output report.html

# Generate DNA fingerprint for a single dataset
datadna fingerprint real_data.csv --output fingerprint.json

# Benchmark multiple synthetic datasets
datadna benchmark real_data.csv syn1.csv syn2.csv syn3.csv --output benchmark.html
```

## DNA Fingerprint Structure

The DNA string encodes 12 statistical properties into a 48-character hex string:

```
Position 1-4:   Column count entropy hash
Position 5-8:   Row count magnitude encoding
Position 9-12:  Mean distribution fingerprint
Position 13-16: Variance profile signature
Position 17-20: Skewness pattern code
Position 21-24: Kurtosis indicators
Position 25-28: Correlation matrix hash
Position 29-32: Missing data pattern
Position 33-36: Outlier density signature
Position 37-40: Category distribution hash
Position 41-44: Temporal pattern code (if time-series)
Position 45-48: Entropy fingerprint
```

Example: `3FA7-B2C1-8E4D-91A6-5C3F-D7E2-A8B4-6F1D-E9C3-2A58-D4B7-1F6E`

## Quality Metrics

| Metric | Weight | Description |
|---|---|---|
| Distribution Fidelity | 20% | KS-test + Wasserstein distance per column |
| Correlation Preservation | 15% | Frobenius norm of correlation matrix difference |
| Outlier Fidelity | 12% | Mahalanobis distance overlap |
| Entropy Match | 10% | Mutual information between feature distributions |
| Category Balance | 8% | Jensen-Shannon divergence for categorical columns |
| ML Utility | 15% | Cross-evaluation accuracy ratio |
| Mode Coverage | 10% | Mode collapse detection score |
| Statistical Moments | 5% | Mean, variance, skew, kurtosis preservation |
| Missing Pattern | 3% | Missing data pattern similarity |
| Temporal Consistency | 2% | Autocorrelation preservation (time-series) |

## Output

```
┌─────────────────────────────────────────────────┐
│  DataDNA Analysis Report                        │
├─────────────────────────────────────────────────┤
│  Real Dataset:      10,000 rows × 25 columns   │
│  Synthetic Dataset: 10,000 rows × 25 columns   │
├─────────────────────────────────────────────────┤
│  Quality Score: 87/100  █████████████████░░     │
│  Grade: A                                     │
│  Mode Collapse: None detected                  │
│  ML Utility: 92% (train on syn, test on real)  │
├─────────────────────────────────────────────────┤
│  Real DNA:  3FA7-B2C1-8E4D-91A6-...           │
│  Syn DNA:   3FA9-B2C0-8E4D-91A7-...           │
│  Match:     94.2% similarity                   │
└─────────────────────────────────────────────────┘
```

## License

MIT License
