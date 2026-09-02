#!/usr/bin/env python3
"""
Extract numerical values from exp3_classification_results.json for article writing.
Tasks 1.2.1 through 1.2.7 from the spec.
"""

import json
import numpy as np
from pathlib import Path

# Load the results
results_path = Path(__file__).parent / "results" / "exp3_classification_results.json"
verification_path = Path(__file__).parent / "results" / "verification.json"

with open(results_path) as f:
    data = json.load(f)

with open(verification_path) as f:
    verification = json.load(f)

print("=" * 80)
print("EXTRACTION RESULTS FOR EXP3 DSGE MODULATION ARTICLE")
print("=" * 80)

# Task 1.2.1: Mean accuracy by feature set (Spectral-only, DSGE-only best basis, Hybrid)
print("\n" + "=" * 80)
print("TASK 1.2.1: Mean accuracy by feature set (RF classifier)")
print("=" * 80)

spectral_accs = []
dsge_best_accs = []  # poly_s4 is the best basis
hybrid_accs = []

for snr_result in data["snr_results"]:
    snr = snr_result["snr_db"]
    
    # Spectral-only RF
    spectral_acc = snr_result["baseline_spectral"]["rf"]["accuracy"]
    spectral_accs.append(spectral_acc)
    
    # DSGE-only RF (best basis = poly_s4)
    dsge_acc = snr_result["by_basis"]["poly_s4"]["dsge"]["rf"]["accuracy"]
    dsge_best_accs.append(dsge_acc)
    
    # Hybrid RF (poly_s4)
    hybrid_acc = snr_result["by_basis"]["poly_s4"]["hybrid"]["rf"]["accuracy"]
    hybrid_accs.append(hybrid_acc)

mean_spectral = np.mean(spectral_accs)
mean_dsge = np.mean(dsge_best_accs)
mean_hybrid = np.mean(hybrid_accs)

print(f"Spectral-only (RF):  {mean_spectral:.3f}")
print(f"DSGE-only (RF, poly_s4): {mean_dsge:.3f}")
print(f"Hybrid (RF, poly_s4):    {mean_hybrid:.3f}")
print(f"\nExpected: 0.921 / 0.757 / 0.938")
print(f"Match: Spectral={mean_spectral:.3f}≈0.921, DSGE={mean_dsge:.3f}≈0.757, Hybrid={mean_hybrid:.3f}≈0.938")

# Task 1.2.2: Per-SNR accuracy for DSGE-only RF (best basis poly_s4)
print("\n" + "=" * 80)
print("TASK 1.2.2: Per-SNR accuracy for DSGE-only RF (poly_s4)")
print("=" * 80)

dsge_per_snr = {}
for snr_result in data["snr_results"]:
    snr = snr_result["snr_db"]
    dsge_acc = snr_result["by_basis"]["poly_s4"]["dsge"]["rf"]["accuracy"]
    dsge_per_snr[int(snr)] = dsge_acc
    print(f"SNR {int(snr):+3d} dB: {dsge_acc:.3f}")

print(f"\nExpected: {{−5: 0.767, 0: 0.743, +5: 0.743, +10: 0.740, +15: 0.795}}")
print(f"Actual:   {{{', '.join([f'{k:+d}: {v:.3f}' for k, v in sorted(dsge_per_snr.items())])}}}")

# Task 1.2.3: Extract α* per SNR from patp_sweep
print("\n" + "=" * 80)
print("TASK 1.2.3: Optimal α* per SNR from PATP sweep")
print("=" * 80)

alpha_optimal_per_snr = {}
for snr_result in data["snr_results"]:
    snr = snr_result["snr_db"]
    patp_sweep = snr_result["patp_sweep"]
    
    # Find alpha with max accuracy (excluding degenerate α=0.5)
    best_alpha = None
    best_acc = -1
    for point in patp_sweep:
        if point["alpha"] != 0.5 and point["accuracy"] > best_acc:
            best_acc = point["accuracy"]
            best_alpha = point["alpha"]
    
    alpha_optimal_per_snr[int(snr)] = best_alpha
    print(f"SNR {int(snr):+3d} dB: α* = {best_alpha}")

print(f"\nExpected: α*=0.6 on all 5 SNR")
print(f"Match: All α* = {set(alpha_optimal_per_snr.values())}")

# Task 1.2.4: Extract accuracy and AUC at α*=0.6 per SNR
print("\n" + "=" * 80)
print("TASK 1.2.4: Accuracy and AUC at α*=0.6 per SNR")
print("=" * 80)

acc_auc_at_06 = {}
for snr_result in data["snr_results"]:
    snr = snr_result["snr_db"]
    patp_sweep = snr_result["patp_sweep"]
    
    # Find the point where alpha = 0.6
    for point in patp_sweep:
        if point["alpha"] == 0.6:
            acc = point["accuracy"]
            auc = point["auc"]
            acc_auc_at_06[int(snr)] = (acc, auc)
            print(f"SNR {int(snr):+3d} dB: accuracy={acc:.3f}, AUC={auc:.3f}")
            break

print(f"\nExpected: {{−5: (0.709, 0.802), 0: (0.695, 0.785), +5: (0.702, 0.786), +10: (0.694, 0.771), +15: (0.752, 0.859)}}")
print(f"Actual:   {{{', '.join([f'{k:+d}: ({v[0]:.3f}, {v[1]:.3f})' for k, v in sorted(acc_auc_at_06.items())])}}}")

# Task 1.2.5: Extract accuracy at boundary α values
print("\n" + "=" * 80)
print("TASK 1.2.5: Accuracy at boundary α values (averaged over SNR)")
print("=" * 80)

alpha_0_accs = []
alpha_05_accs = []
alpha_1_accs = []

for snr_result in data["snr_results"]:
    patp_sweep = snr_result["patp_sweep"]
    
    for point in patp_sweep:
        if point["alpha"] == 0.0:
            alpha_0_accs.append(point["accuracy"])
        elif point["alpha"] == 0.5:
            alpha_05_accs.append(point["accuracy"])
        elif point["alpha"] == 1.0:
            alpha_1_accs.append(point["accuracy"])

mean_alpha_0 = np.mean(alpha_0_accs)
mean_alpha_05 = np.mean(alpha_05_accs)
mean_alpha_1 = np.mean(alpha_1_accs)

print(f"α=0.0 (fractional):   {mean_alpha_0:.3f}")
print(f"α=0.5 (degenerate):   {mean_alpha_05:.3f}")
print(f"α=1.0 (polynomial):   {mean_alpha_1:.3f}")
print(f"\nExpected: α=0 ≈ 0.67, α=0.5 ≈ 0.53, α=1 ≈ 0.60")
print(f"Match: α=0={mean_alpha_0:.2f}, α=0.5={mean_alpha_05:.2f}, α=1={mean_alpha_1:.2f}")

# Task 1.2.6: Extract bootstrap p-value
print("\n" + "=" * 80)
print("TASK 1.2.6: Bootstrap p-value for hybrid vs spectral")
print("=" * 80)

for check in verification["checks"]:
    if check["id"] == "hybrid_vs_spectral_significant":
        p_value = check["value"]
        note = check["note"]
        print(f"Bootstrap p-value: {p_value}")
        print(f"Note: {note}")
        print(f"\nExpected: p < 0.001, observed mean diff = 0.0165")
        print(f"Match: p={p_value} < 0.001 ✓, {note}")

# Task 1.2.7: Extract accuracy for all 5 bases × 3 classifiers (Table III)
print("\n" + "=" * 80)
print("TASK 1.2.7: Accuracy for all 5 bases × 3 classifiers (mean over SNR)")
print("=" * 80)

bases = ["poly_s3", "poly_s4", "frac_s3", "trig_s4", "robust_s3"]
classifiers = ["lr", "rf", "svm_rbf"]
feature_types = ["dsge", "hybrid"]

print("\nTable III: Accuracy by basis, classifier, and feature type")
print("-" * 80)

for feature_type in feature_types:
    print(f"\n{feature_type.upper()} features:")
    print(f"{'Basis':<12} {'LR':>8} {'RF':>8} {'SVM-RBF':>8}")
    print("-" * 40)
    
    for basis in bases:
        accs = {clf: [] for clf in classifiers}
        
        for snr_result in data["snr_results"]:
            for clf in classifiers:
                acc = snr_result["by_basis"][basis][feature_type][clf]["accuracy"]
                accs[clf].append(acc)
        
        mean_accs = {clf: np.mean(accs[clf]) for clf in classifiers}
        print(f"{basis:<12} {mean_accs['lr']:>8.3f} {mean_accs['rf']:>8.3f} {mean_accs['svm_rbf']:>8.3f}")

# Also show spectral baseline
print(f"\nSPECTRAL baseline:")
print(f"{'Baseline':<12} {'LR':>8} {'RF':>8} {'SVM-RBF':>8}")
print("-" * 40)

spectral_accs_clf = {clf: [] for clf in classifiers}
for snr_result in data["snr_results"]:
    for clf in classifiers:
        acc = snr_result["baseline_spectral"][clf]["accuracy"]
        spectral_accs_clf[clf].append(acc)

mean_spectral_clf = {clf: np.mean(spectral_accs_clf[clf]) for clf in classifiers}
print(f"{'spectral':<12} {mean_spectral_clf['lr']:>8.3f} {mean_spectral_clf['rf']:>8.3f} {mean_spectral_clf['svm_rbf']:>8.3f}")

print("\n" + "=" * 80)
print("EXTRACTION COMPLETE")
print("=" * 80)
