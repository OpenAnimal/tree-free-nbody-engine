"""
Module 13: Polygenic Mendelian Randomization (MR) & Instrumental Variable Causal Inference Engine.
Estimates unconfounded causal effect of exposure biomolecules (e.g. LDL cholesterol, IL-6 cytokine, drug targets)
on clinical disease outcomes (e.g. Coronary Artery Disease, Alzheimer's) using genetic variant instruments (SNPs).
Implements Inverse-Variance Weighted (IVW), MR-Egger pleiotropy tests, and Weighted Median estimators.
"""

from __future__ import annotations
import numpy as np
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class GeneticInstrument:
    """Genetic single nucleotide polymorphism (SNP) instrument."""
    snp_id: str                   # e.g., "rs12740374"
    chromosome: str
    position_bp: int
    effect_allele: str
    beta_exposure: float          # Effect size on exposure (gamma_j)
    se_exposure: float            # Standard error of gamma_j
    beta_outcome: float           # Effect size on outcome (Gamma_j)
    se_outcome: float             # Standard error of Gamma_j
    f_statistic: float            # Instrument strength (F > 10 avoids weak instrument bias)


@dataclass
class MendelianRandomizationReport:
    """Comprehensive causal effect report derived from polygenic Mendelian Randomization."""
    exposure_name: str            # e.g., "Circulating_LDL_C"
    outcome_name: str             # e.g., "Coronary_Artery_Disease"
    num_instruments_used: int
    mean_f_statistic: float
    causal_effect_ivw: float      # Inverse-Variance Weighted beta
    se_ivw: float
    p_value_ivw: float
    causal_effect_egger: float    # MR-Egger slope
    egger_pleiotropy_intercept: float # Egger intercept alpha (tests directional horizontal pleiotropy)
    egger_intercept_p_value: float    # p < 0.05 indicates presence of pleiotropy bias
    causal_effect_weighted_median: float
    cochran_q_heterogeneity_p_value: float # Test for variant heterogeneity
    causal_conclusion: str        # "Statistically Significant Causal Effect", "No Causal Evidence", "Pleiotropy Bias Detected"


class PolygenicMendelianRandomizationEngine:
    """
    High-Throughput Polygenic Mendelian Randomization Engine for Drug Target Validation & Epidemiology.
    Uses nature's randomized genetic allocation at meiosis to establish unconfounded causality.
    """
    def __init__(self, f_stat_filter: float = 10.0):
        self.f_stat_filter = float(f_stat_filter)
        if not np.isfinite(self.f_stat_filter) or self.f_stat_filter <= 0.0:
            raise ValueError("f_stat_filter must be finite and positive")

    def estimate_causal_effect(
        self,
        instruments: List[GeneticInstrument],
        exposure_name: str = "Biomarker_Exposure",
        outcome_name: str = "Clinical_Outcome"
    ) -> MendelianRandomizationReport:
        """
        Calculates causal effect beta across polygenic instruments using IVW, MR-Egger, and Weighted Median.
        """
        # 1. Filter weak instruments (F-statistic >= 10.0) and malformed estimates
        valid_snps = [
            snp for snp in instruments
            if np.isfinite(snp.f_statistic) and snp.f_statistic >= self.f_stat_filter
            and np.isfinite(snp.beta_exposure) and abs(snp.beta_exposure) > 1e-12
            and np.isfinite(snp.beta_outcome)
            and np.isfinite(snp.se_outcome) and snp.se_outcome > 0.0
            and np.isfinite(snp.se_exposure) and snp.se_exposure > 0.0
        ]
        if len(valid_snps) < 2:
            raise ValueError(f"Need at least 2 strong genetic instruments (F >= {self.f_stat_filter}), found {len(valid_snps)}.")

        gamma = np.array([snp.beta_exposure for snp in valid_snps], dtype=np.float64) # (J,)
        Gamma = np.array([snp.beta_outcome for snp in valid_snps], dtype=np.float64)  # (J,)
        se_Gamma = np.array([snp.se_outcome for snp in valid_snps], dtype=np.float64) # (J,)
        f_stats = np.array([snp.f_statistic for snp in valid_snps], dtype=np.float64)

        J = len(valid_snps)

        # 2. Inverse-Variance Weighted (IVW) Estimator
        # Wald ratio per SNP: ratio_j = Gamma_j / gamma_j
        # Weight per SNP: w_j = (gamma_j^2) / (se_Gamma_j^2)
        ratio_estimates = Gamma / gamma
        weights_ivw = (gamma ** 2) / (se_Gamma ** 2)

        beta_ivw = float(np.sum(weights_ivw * ratio_estimates) / np.sum(weights_ivw))
        se_ivw = float(np.sqrt(1.0 / np.sum(weights_ivw)))
        z_ivw = beta_ivw / max(1e-9, se_ivw)
        p_val_ivw = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_ivw) / np.sqrt(2.0)))))

        # Cochran's Q statistic for heterogeneity: Q = sum w_j * (ratio_j - beta_ivw)^2
        q_stat = float(np.sum(weights_ivw * ((ratio_estimates - beta_ivw) ** 2)))
        # Chi-squared p-value approximation with J - 1 degrees of freedom
        q_p_val = float(np.exp(-0.5 * max(0.0, q_stat - (J - 1))))

        # 3. MR-Egger Regression (Gamma_j = beta_0 + beta_egger * gamma_j)
        # Weighted by 1 / se_Gamma_j^2
        w_egger = 1.0 / (se_Gamma ** 2)
        w_norm = w_egger / np.sum(w_egger)

        # Weighted linear regression: X = gamma, Y = Gamma
        x_mean = np.sum(w_norm * gamma)
        y_mean = np.sum(w_norm * Gamma)
        cov_xy = np.sum(w_norm * (gamma - x_mean) * (Gamma - y_mean))
        var_x = np.sum(w_norm * ((gamma - x_mean) ** 2))

        if var_x > 1e-12:
            beta_egger = float(cov_xy / var_x)
            alpha_egger = float(y_mean - beta_egger * x_mean)
        else:
            beta_egger = beta_ivw
            alpha_egger = 0.0

        # Test for directional pleiotropy (H0: alpha == 0)
        # Correct SE of the intercept in weighted least squares:
        #   Var(alpha_hat) = 1/sum(w) + x_mean^2 / sum(w * (x - x_mean)^2)
        # The previous formula sqrt(1/sum(w)) * 0.8 had an arbitrary 0.8
        # fudge factor and omitted the x_mean^2 / var_x term, giving a
        # biased (typically too small) intercept SE and inflated
        # pleiotropy significance.
        if var_x > 1e-12:
            se_alpha = float(np.sqrt(1.0 / np.sum(w_egger) + (x_mean ** 2) / (np.sum(w_egger) * var_x)))
        else:
            se_alpha = float(np.sqrt(1.0 / np.sum(w_egger)))
        z_alpha = alpha_egger / max(1e-9, se_alpha)
        p_val_alpha = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_alpha) / np.sqrt(2.0)))))

        # 4. Weighted Median Estimator
        order = np.argsort(ratio_estimates)
        sorted_ratios = ratio_estimates[order]
        sorted_weights = weights_ivw[order] / np.sum(weights_ivw)
        cum_weights = np.cumsum(sorted_weights)
        med_idx = np.where(cum_weights >= 0.5)[0][0]
        beta_wm = float(sorted_ratios[med_idx])

        # 5. Conclusion synthesis
        if p_val_alpha < 0.05:
            conclusion = "Pleiotropy Bias Detected (Rely on MR-Egger / Weighted Median)"
        elif p_val_ivw < 0.05:
            conclusion = "Statistically Significant Causal Effect"
        else:
            conclusion = "No Significant Causal Evidence (Null Association)"

        return MendelianRandomizationReport(
            exposure_name=exposure_name,
            outcome_name=outcome_name,
            num_instruments_used=J,
            mean_f_statistic=float(np.mean(f_stats)),
            causal_effect_ivw=beta_ivw,
            se_ivw=se_ivw,
            p_value_ivw=p_val_ivw,
            causal_effect_egger=beta_egger,
            egger_pleiotropy_intercept=alpha_egger,
            egger_intercept_p_value=p_val_alpha,
            causal_effect_weighted_median=beta_wm,
            cochran_q_heterogeneity_p_value=q_p_val,
            causal_conclusion=conclusion
        )
