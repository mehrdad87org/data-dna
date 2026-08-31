import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from datadna.metrics import FidelityMetrics


@dataclass
class DNAFingerprint:
    column_count_entropy: float = 0.0
    row_count_magnitude: float = 0.0
    mean_distribution: str = ""
    variance_profile: str = ""
    skewness_pattern: str = ""
    kurtosis_indicators: str = ""
    correlation_hash: str = ""
    missing_pattern: str = ""
    outlier_density: str = ""
    category_hash: str = ""
    temporal_pattern: str = ""
    entropy_fingerprint: str = ""

    def to_hex_string(self) -> str:
        parts = [
            self.column_count_entropy,
            self.row_count_magnitude,
            self.mean_distribution,
            self.variance_profile,
            self.skewness_pattern,
            self.kurtosis_indicators,
            self.correlation_hash,
            self.missing_pattern,
            self.outlier_density,
            self.category_hash,
            self.temporal_pattern,
            self.entropy_fingerprint,
        ]
        hex_strs = []
        for p in parts:
            if isinstance(p, float):
                h = hashlib.md5(str(p).encode()).hexdigest()[:4].upper()
            else:
                h = hashlib.md5(str(p).encode()).hexdigest()[:4].upper()
            hex_strs.append(h)
        return "-".join(hex_strs)


@dataclass
class DNAComparison:
    quality_score: float = 0.0
    grade: str = ""
    mode_collapse: bool = False
    ml_utility_score: float = 0.0
    dna_similarity: float = 0.0
    real_dna: str = ""
    synthetic_dna: str = ""
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    per_column_scores: Dict[str, float] = field(default_factory=dict)
    mode_collapse_details: Dict = field(default_factory=dict)
    real_shape: Tuple[int, int] = (0, 0)
    synthetic_shape: Tuple[int, int] = (0, 0)

    def generate_report(self, output_path: str = "datadna_report.html"):
        from datadna.report import generate_html_report
        generate_html_report(self, output_path)


class DataDNA:
    def __init__(self, data: pd.DataFrame, name: str = "dataset"):
        self.data = data
        self.name = name
        self.metrics = FidelityMetrics()
        self.fingerprint = self._compute_fingerprint()
        self.dna_string = self.fingerprint.to_hex_string()

    def _compute_fingerprint(self) -> DNAFingerprint:
        fp = DNAFingerprint()
        data = self.data

        fp.column_count_entropy = self._entropy_hash(len(data.columns))
        fp.row_count_magnitude = np.log10(max(len(data), 1)) / 10

        numeric = data.select_dtypes(include=[np.number])
        if not numeric.empty:
            means = numeric.mean()
            fp.mean_distribution = hashlib.md5(
                str(means.values.round(4)).encode()
            ).hexdigest()[:8]

            variances = numeric.var()
            fp.variance_profile = hashlib.md5(
                str(variances.values.round(4)).encode()
            ).hexdigest()[:8]

            from scipy import stats as sp_stats
            skews = numeric.apply(lambda x: sp_stats.skew(x.dropna()))
            fp.skewness_pattern = hashlib.md5(
                str(skews.values.round(4)).encode()
            ).hexdigest()[:8]

            kurtoses = numeric.apply(lambda x: sp_stats.kurtosis(x.dropna()))
            fp.kurtosis_indicators = hashlib.md5(
                str(kurtoses.values.round(4)).encode()
            ).hexdigest()[:8]

            corr = numeric.corr().fillna(0).values
            fp.correlation_hash = hashlib.md5(
                str(corr.round(4)).encode()
            ).hexdigest()[:8]

            outlier_counts = []
            for col in numeric.columns:
                q1, q3 = numeric[col].quantile(0.25), numeric[col].quantile(0.75)
                iqr = q3 - q1
                outliers = ((numeric[col] < q1 - 1.5 * iqr) | (numeric[col] > q3 + 1.5 * iqr)).sum()
                outlier_counts.append(outliers)
            fp.outlier_density = hashlib.md5(str(outlier_counts).encode()).hexdigest()[:8]

            entropies = numeric.apply(lambda x: self._hist_entropy(x.dropna()))
            fp.entropy_fingerprint = hashlib.md5(
                str(entropies.values.round(4)).encode()
            ).hexdigest()[:8]
        else:
            fp.mean_distribution = "0"
            fp.variance_profile = "0"
            fp.skewness_pattern = "0"
            fp.kurtosis_indicators = "0"
            fp.correlation_hash = "0"
            fp.outlier_density = "0"
            fp.entropy_fingerprint = "0"

        missing = data.isna().mean()
        fp.missing_pattern = hashlib.md5(str(missing.values.round(4)).encode()).hexdigest()[:8]

        cat_cols = data.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            cat_hashes = []
            for col in cat_cols:
                vc = data[col].value_counts(normalize=True)
                cat_hashes.append(hashlib.md5(str(vc.values.round(4)).encode()).hexdigest()[:4])
            fp.category_hash = hashlib.md5("".join(cat_hashes).encode()).hexdigest()[:8]
        else:
            fp.category_hash = "0"

        date_cols = data.select_dtypes(include=["datetime64"]).columns
        if len(date_cols) > 0:
            fp.temporal_pattern = hashlib.md5(
                str([data[c].std() for c in date_cols]).encode()
            ).hexdigest()[:8]
        else:
            fp.temporal_pattern = "0"

        return fp

    def _entropy_hash(self, value: int) -> float:
        return round(np.log10(max(value, 1)) / 5, 4)

    def _hist_entropy(self, values: pd.Series) -> float:
        if len(values) < 2:
            return 0.0
        n_bins = min(30, max(5, int(np.sqrt(len(values)))))
        hist, _ = np.histogram(values, bins=n_bins, density=True)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        return float(np.sum(-hist * np.log2(hist)))

    def compare(self, other: "DataDNA") -> DNAComparison:
        comparison = DNAComparison()
        comparison.real_dna = self.dna_string
        comparison.synthetic_dna = other.dna_string
        comparison.real_shape = self.data.shape
        comparison.synthetic_shape = other.data.shape

        scores = {}

        scores["distribution_fidelity"] = self._average_metric(
            self.metrics.distribution_fidelity(self.data, other.data)
        )
        scores["correlation_preservation"] = self.metrics.correlation_preservation(self.data, other.data)
        scores["outlier_fidelity"] = self.metrics.outlier_fidelity(self.data, other.data)
        scores["entropy_match"] = self.metrics.entropy_match(self.data, other.data)
        scores["category_balance"] = self.metrics.category_balance(self.data, other.data)
        scores["statistical_moments"] = self.metrics.statistical_moments(self.data, other.data)
        scores["missing_pattern"] = self.metrics.missing_pattern(self.data, other.data)

        mc = self.metrics.mode_collapse_detection(self.data, other.data)
        scores["mode_coverage"] = mc["average_coverage"]
        comparison.mode_collapse_details = mc
        comparison.mode_collapse = mc["mode_collapse_detected"]

        scores["ml_utility"] = self.metrics.ml_utility(self.data, other.data)

        weights = {
            "distribution_fidelity": 0.20,
            "correlation_preservation": 0.15,
            "outlier_fidelity": 0.12,
            "entropy_match": 0.10,
            "category_balance": 0.08,
            "mode_coverage": 0.10,
            "ml_utility": 0.15,
            "statistical_moments": 0.05,
            "missing_pattern": 0.03,
        }

        weighted_score = sum(scores.get(k, 0.5) * v for k, v in weights.items())
        comparison.quality_score = round(weighted_score * 100, 1)
        comparison.ml_utility_score = round(scores.get("ml_utility", 0.5) * 100, 1)

        dna_sim = self._dna_similarity(self.dna_string, other.dna_string)
        comparison.dna_similarity = round(dna_sim, 4)

        if comparison.quality_score >= 90:
            comparison.grade = "A+"
        elif comparison.quality_score >= 80:
            comparison.grade = "A"
        elif comparison.quality_score >= 70:
            comparison.grade = "B+"
        elif comparison.quality_score >= 60:
            comparison.grade = "B"
        elif comparison.quality_score >= 50:
            comparison.grade = "C+"
        elif comparison.quality_score >= 40:
            comparison.grade = "C"
        else:
            comparison.grade = "D"

        comparison.dimension_scores = {k: round(v, 4) for k, v in scores.items()}
        comparison.per_column_scores = self.metrics.distribution_fidelity(self.data, other.data)

        return comparison

    def _average_metric(self, scores_dict: Dict[str, float]) -> float:
        if not scores_dict:
            return 0.5
        return list(scores_dict.values())[0] if len(scores_dict) == 1 else np.mean(list(scores_dict.values()))

    def _dna_similarity(self, dna1: str, dna2: str) -> float:
        parts1 = dna1.split("-")
        parts2 = dna2.split("-")
        if len(parts1) != len(parts2):
            return 0.0
        matches = sum(1 for a, b in zip(parts1, parts2) if a == b)
        return matches / len(parts1)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "shape": self.data.shape,
            "dna_string": self.dna_string,
            "fingerprint": asdict(self.fingerprint),
            "column_types": {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            "numeric_summary": {
                col: {
                    "mean": round(float(self.data[col].mean()), 4),
                    "std": round(float(self.data[col].std()), 4),
                    "min": round(float(self.data[col].min()), 4),
                    "max": round(float(self.data[col].max()), 4),
                }
                for col in self.data.select_dtypes(include=[np.number]).columns
            },
        }
