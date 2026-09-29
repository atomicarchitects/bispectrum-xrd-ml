import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 14

bispec_file = os.environ.get('POWDERXRD_DATA_ROOT', './data') + '/materials_project_lat_bispec.h5'

print("Loading bispectrum data...")
with h5py.File(bispec_file, 'r') as f:
    bispecs = f["bispec"][:]  # (n_materials, n_coeffs, ...)

print(f"Loaded {bispecs.shape[0]} materials, shape: {bispecs.shape}")

# Coefficients with |b| <= this are treated as zero everywhere else in the code (the training loss mask and the
# structured head use mask_loss_threshold = 1e-10), so the same threshold is used here.
MASK_THRESHOLD = 1e-10

vals = bispecs.flatten()
n_total = vals.size
vals = vals[np.abs(vals) > MASK_THRESHOLD]
print(f"Total coefficients: {n_total:,}")
print(f"Non-zero coefficients (|b| > {MASK_THRESHOLD:g}): {len(vals):,} ({100 * len(vals) / n_total:.1f}%)")

fig, ax = plt.subplots(figsize=(6, 4))

# |b_i| on log x-axis to show dynamic range
abs_vals_plot = np.abs(vals)
log_bins = np.logspace(np.log10(abs_vals_plot.min()), np.log10(abs_vals_plot.max()), 200)
ax.hist(abs_vals_plot, bins=log_bins, color='steelblue', alpha=0.8)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$|b_i|$')
ax.set_ylabel('Count')
ax.set_title('Bispectrum coefficient magnitudes')

plt.tight_layout()
plt.savefig('bispec_distribution.pdf', bbox_inches='tight')
plt.savefig('bispec_distribution.png', dpi=150, bbox_inches='tight')
print("Saved bispec_distribution.pdf and .png")

# Print summary stats to support the claim
print(f"\n--- Raw coefficient stats (|b| > {MASK_THRESHOLD:g}) ---")
abs_vals = np.abs(vals)
print(f"  min:    {abs_vals.min():.3e}")
print(f"  median: {np.median(abs_vals):.3e}")
print(f"  mean:   {abs_vals.mean():.3e}")
print(f"  p95:    {np.percentile(abs_vals, 95):.3e}")
print(f"  p99:    {np.percentile(abs_vals, 99):.3e}")
print(f"  max:    {abs_vals.max():.3e}")
print(f"  mean/median ratio: {abs_vals.mean()/np.median(abs_vals):.1f}x  (>>1 => heavy tail)")
