import hashlib
import math
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import mahalanobis, jensenshannon
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import mutual_info_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


class FidelityMetrics:
    """Computes statistical fidelity metrics between real and synthetic datasets."""

    @staticmethod
    def distribution_fidelity(real: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, float]:
        scores = {}
        numeric_cols = real.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col not in synthetic.columns:
                continue
            real_vals = real[col].dropna().values
            syn_vals = synthetic[col].dropna().values
            if len(real_vals) < 2 or len(syn_vals) < 2:
                continue

            ks_stat, _ = stats.ks_2samp(real_vals, syn_vals)
            wasserstein = stats.wasserstein_distance(real_vals, syn_vals)
            real_range = real_vals.max() - real_vals.min()
            if real_range > 0:
                normalized_wasserstein = wasserstein / real_range
            else:
                normalized_wasserstein = 0

            score = max(0, 1 - ks_stat) * 0.5 + max(0, 1 - normalized_wasserstein) * 0.5
            scores[col] = round(score, 4)

        categorical_cols = real.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            if col not in synthetic.columns:
                continue
            real_counts = real[col].value_counts(normalize=True).sort_index()
            syn_counts = synthetic[col].value_counts(normalize=True).sort_index()
            all_cats = sorted(set(real_counts.index) | set(syn_counts.index))
            real_dist = np.array([real_counts.get(c, 0) for c in all_cats])
            syn_dist = np.array([syn_counts.get(c, 0) for c in all_cats])
            if len(all_cats) > 1:
                js_div = jensenshannon(real_dist, syn_dist)
                score = max(0, 1 - js_div)
            else:
                score = 1.0
            scores[col] = round(score, 4)

        return scores

    @staticmethod
    def correlation_preservation(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        numeric_real = real.select_dtypes(include=[np.number])
        numeric_syn = synthetic.select_dtypes(include=[np.number])
        common_cols = list(set(numeric_real.columns) & set(numeric_syn.columns))

        if len(common_cols) < 2:
            return 1.0

        real_corr = numeric_real[common_cols].corr().fillna(0).values
        syn_corr = numeric_syn[common_cols].corr().fillna(0).values
        frobenius = np.linalg.norm(real_corr - syn_corr)
        n = len(common_cols)
        max_frobenius = n * np.sqrt(2)
        if max_frobenius > 0:
            score = max(0, 1 - frobenius / max_frobenius)
        else:
            score = 1.0
        return round(score, 4)

    @staticmethod
    def outlier_fidelity(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        numeric_real = real.select_dtypes(include=[np.number]).dropna()
        numeric_syn = synthetic.select_dtypes(include=[np.number]).dropna()
        common_cols = list(set(numeric_real.columns) & set(numeric_syn.columns))

        if len(common_cols) < 2 or len(numeric_real) < 10:
            return 1.0

        real_q1 = numeric_real[common_cols].quantile(0.25)
        real_q3 = numeric_real[common_cols].quantile(0.75)
        iqr = real_q3 - real_q1
        lower = real_q1 - 1.5 * iqr
        upper = real_q3 + 1.5 * iqr

        real_outliers = ((numeric_real[common_cols] < lower) | (numeric_real[common_cols] > upper)).any(axis=1).sum()
        syn_outliers = ((numeric_syn[common_cols] < lower) | (numeric_syn[common_cols] > upper)).any(axis=1).sum()

        real_ratio = real_outliers / max(len(numeric_real), 1)
        syn_ratio = syn_outliers / max(len(numeric_syn), 1)

        if real_ratio > 0:
            ratio_diff = abs(real_ratio - syn_ratio)
            score = max(0, 1 - ratio_diff / real_ratio)
        else:
            score = 1.0 if syn_ratio == 0 else 0.8

        return round(score, 4)

    @staticmethod
    def entropy_match(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        scores = []
        numeric_cols = real.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col not in synthetic.columns:
                continue
            real_vals = real[col].dropna()
            syn_vals = synthetic[col].dropna()
            if len(real_vals) < 5 or len(syn_vals) < 5:
                continue

            n_bins = min(30, max(5, int(np.sqrt(len(real_vals)))))
            real_hist, edges = np.histogram(real_vals, bins=n_bins, density=True)
            syn_hist, _ = np.histogram(syn_vals, bins=edges, density=True)

            real_hist = real_hist + 1e-10
            syn_hist = syn_hist + 1e-10
            real_hist = real_hist / real_hist.sum()
            syn_hist = syn_hist / syn_hist.sum()

            mi = mutual_info_score(
                np.digitize(real_vals, edges),
                np.digitize(real_vals, edges)
            )
            re = scipy_entropy(real_hist)
            if re > 0:
                score = min(1.0, mi / re)
            else:
                score = 1.0
            scores.append(score)

        return round(np.mean(scores) if scores else 1.0, 4)

    @staticmethod
    def category_balance(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        scores = []
        cat_cols = real.select_dtypes(include=["object", "category"]).columns

        for col in cat_cols:
            if col not in synthetic.columns:
                continue
            real_counts = real[col].value_counts(normalize=True).sort_index()
            syn_counts = synthetic[col].value_counts(normalize=True).sort_index()
            all_cats = sorted(set(real_counts.index) | set(syn_counts.index))
            if len(all_cats) < 2:
                continue

            real_dist = np.array([real_counts.get(c, 0) for c in all_cats])
            syn_dist = np.array([syn_counts.get(c, 0) for c in all_cats])
            js = jensenshannon(real_dist, syn_dist)
            scores.append(max(0, 1 - js))

        return round(np.mean(scores) if scores else 1.0, 4)

    @staticmethod
    def ml_utility(real: pd.DataFrame, synthetic: pd.DataFrame, target_col: Optional[str] = None) -> float:
        numeric_cols = real.select_dtypes(include=[np.number]).columns.tolist()
        if target_col is None:
            if "target" in numeric_cols:
                target_col = "target"
            elif len(numeric_cols) > 0:
                target_col = numeric_cols[-1]
            else:
                return 0.5

        if target_col not in real.columns or target_col not in synthetic.columns:
            return 0.5

        feature_cols = [c for c in numeric_cols if c != target_col]
        if not feature_cols:
            return 0.5

        real_clean = real[feature_cols + [target_col]].dropna()
        syn_clean = synthetic[feature_cols + [target_col]].dropna()

        if len(real_clean) < 20 or len(syn_clean) < 20:
            return 0.5

        X_real = real_clean[feature_cols].values
        y_real = real_clean[target_col].values
        X_syn = syn_clean[feature_cols].values
        y_syn = syn_clean[target_col].values

        le = LabelEncoder()
        y_real_enc = le.fit_transform(y_real)
        y_syn_enc = le.fit_transform(y_syn)

        clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)

        try:
            scores_real = cross_val_score(clf, X_real, y_real_enc, cv=min(5, len(real_clean) // 4), scoring="accuracy")
            real_acc = scores_real.mean()
        except Exception:
            real_acc = 0.5

        try:
            clf.fit(X_syn, y_syn_enc)
            syn_acc = clf.score(X_real, y_real_enc)
        except Exception:
            syn_acc = 0.5

        if real_acc > 0:
            score = min(1.0, syn_acc / real_acc)
        else:
            score = 0.5

        return round(score, 4)

    @staticmethod
    def mode_collapse_detection(real: pd.DataFrame, synthetic: pd.DataFrame) -> Dict:
        numeric_cols = real.select_dtypes(include=[np.number]).columns
        collapse_scores = {}

        for col in numeric_cols:
            if col not in synthetic.columns:
                continue
            real_vals = real[col].dropna().values
            syn_vals = synthetic[col].dropna().values

            real_unique = len(np.unique(real_vals))
            syn_unique = len(np.unique(syn_vals))

            if real_unique > 0:
                coverage = syn_unique / real_unique
                collapse_scores[col] = round(min(1.0, coverage), 4)
            else:
                collapse_scores[col] = 1.0

        avg_coverage = np.mean(list(collapse_scores.values())) if collapse_scores else 1.0
        collapsed_cols = [c for c, v in collapse_scores.items() if v < 0.3]

        return {
            "per_column_coverage": collapse_scores,
            "average_coverage": round(avg_coverage, 4),
            "mode_collapse_detected": len(collapsed_cols) > 0,
            "collapsed_columns": collapsed_cols,
        }

    @staticmethod
    def statistical_moments(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        scores = []
        numeric_cols = real.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col not in synthetic.columns:
                continue
            real_vals = real[col].dropna().values
            syn_vals = synthetic[col].dropna().values

            moments = []
            for i in range(1, 5):
                try:
                    rm = stats.moment(real_vals, moment=i)
                    sm = stats.moment(syn_vals, moment=i)
                    if abs(rm) > 1e-10:
                        diff = abs(rm - sm) / abs(rm)
                        moments.append(max(0, 1 - diff))
                    else:
                        moments.append(1.0 if abs(sm) < 1e-10 else 0.5)
                except Exception:
                    pass

            if moments:
                scores.append(np.mean(moments))

        return round(np.mean(scores) if scores else 1.0, 4)

    @staticmethod
    def missing_pattern(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        common_cols = list(set(real.columns) & set(synthetic.columns))
        if not common_cols:
            return 1.0

        real_missing = [real[c].isna().mean() for c in common_cols]
        syn_missing = [synthetic[c].isna().mean() for c in common_cols]
        diff = np.mean([abs(r - s) for r, s in zip(real_missing, syn_missing)])
        return round(max(0, 1 - diff), 4)
