#%%
import polars as pl

#%%
# Read examples/get_error_and_bpb/bpb_csvs/chunked_rpjv2_sample_bpb_domain.csv
df_bpb = pl.read_csv("get_error_and_bpb/bpb_csvs/chunked_rpjv2_sample_bpb_domain.csv")

# Read examples/get_error_and_bpb/error_csvs/error.csv
df_error = pl.read_csv("get_error_and_bpb/error_csvs/error.csv")

#%%
df_bpb
#%%
df_bpb.columns
"""
['domain',
 "('beespoke', 'BEE-spoke-data/smol_llama-101M-GQA')",
 "('stabilityai', 'stabilityai/stablelm-2-1_6b')",
 "('stabilityai', 'stabilityai/stablelm-3b-4e1t')",
 "('scb10x', 'scb10x/typhoon-7b')",
 "('statespaces', 'state-spaces/mamba-1.4b-hf')",
 "('itsliupeng', 'itsliupeng/openllama-7b-base')",
 "('cyberagent', 'cyberagent/open-calm-large')",
 ...
]"""
# %%
df_error
# %%
df_error.columns
"""
['benchmark',
 "('beespoke', 'BEE-spoke-data/smol_llama-101M-GQA')",
 "('stabilityai', 'stabilityai/stablelm-2-1_6b')",
 "('stabilityai', 'stabilityai/stablelm-3b-4e1t')",
 "('beespoke', 'BEE-spoke-data/smol_llama-81M-tied')",
 "('beespoke', 'BEE-spoke-data/verysmol_llama-v11-KIx2')",
 "('cyberagent', 'cyberagent/open-calm-large')",
 "('statespaces', 'state-spaces/mamba-130m-hf')",
 ...
]
"""
# %%
assert set(df_bpb.columns) - {"domain"} == set(df_error.columns) - {"benchmark"}
# %%
# Sort the columns of df_bpb and df_error (except for the first column)
df_bpb_aligned = df_bpb.select( ["domain"] + sorted(df_bpb.columns[1:]))
df_error_aligned = df_error.select( ["benchmark"] + sorted(df_error.columns[1:]))
#%%
assert df_bpb_aligned.columns[1:] == df_error_aligned.columns[1:]
# %%
df_bpb_aligned
# %%
df_error_aligned
# %%
# Exclude rows in df_error_aligned where the benchmark starts with lambada
# df_error_aligned = df_error_aligned.filter(~pl.col("benchmark").str.starts_with("lambada"))
# %%
# Convert the columns of df_bpb_aligned and df_error_aligned to a dictionary where the key is the first column and the value is a numpy array of the remaining columns
bpb_dict = {row["domain"][0]: row.drop("domain").to_numpy()[0] for row in df_bpb_aligned.iter_slices(n_rows=1)}
error_dict = {row["benchmark"][0]: row.drop("benchmark").to_numpy()[0] for row in df_error_aligned.iter_slices(n_rows=1)}

# %% 
# Compute PMI
import numpy as np
from scipy.special import logsumexp


def compute_pmi(neg_log_prob_x: np.ndarray, neg_log_prob_y: np.ndarray) -> np.ndarray:
    """
    Compute the pointwise mutual information between x and y.
    """
    assert len(neg_log_prob_x) == len(neg_log_prob_y)
    num_samples = len(neg_log_prob_x)
    # Convert to float64
    neg_log_prob_x = neg_log_prob_x.astype(np.float64)
    neg_log_prob_y = neg_log_prob_y.astype(np.float64)
    # Stable computation by trying to minimize issues due to cancellation
    x_min = np.min(neg_log_prob_x)
    y_min = np.min(neg_log_prob_y)
    neg_log_prob_x_marginal = -logsumexp(-neg_log_prob_x + x_min) + np.log(num_samples)
    neg_log_prob_y_marginal = -logsumexp(-neg_log_prob_y + y_min) + np.log(num_samples)
    neg_log_joint_prob_xy = -logsumexp(-neg_log_prob_x - neg_log_prob_y + x_min + y_min) + np.log(num_samples)
    return neg_log_prob_x_marginal + neg_log_prob_y_marginal - neg_log_joint_prob_xy

# %%
# Calculate the Spearman's rank correlation coefficient between the  two dictionaries and compute a new dataframe with the correlation coefficient between domain and benchmark
from scipy.stats import spearmanr
# from perplexity_correlations.estimation import spearmanr

import polars as pl
import numpy as np

# Get the list of models (should be the same for both dataframes after alignment)
models = df_bpb_aligned.columns[1:]

# Create a dataframe to store the correlation results
correlation_results = []

# For each benchmark, calculate correlation with each domain
for benchmark, benchmark_errors in error_dict.items():
    print(benchmark)
    for domain, domain_bpb in bpb_dict.items():
        # Calculate correlation using scipy's spearmanr
        # for bpb and errors, lower is better, so positive correlation is good
        corr, p_value = spearmanr(domain_bpb, benchmark_errors)
        # corr = spearmanr(domain_bpb[:, None], benchmark_errors).item()
        
        if benchmark.startswith("lambada"):
            # Perplexity to cross-entropy
            benchmark_neg_log_prob = np.log(benchmark_errors)
        else:
            # Error rate to NLL
            benchmark_neg_log_prob = -np.log(1.0 - benchmark_errors)
            assert all(benchmark_neg_log_prob >= 0)
        
        pmi = compute_pmi(domain_bpb, benchmark_neg_log_prob)
        
        correlation_results.append({
            "benchmark": benchmark,
            "domain": domain,
            "rank_correlation": corr,
            "pmi": pmi
        })

#%%
# Convert to polars DataFrame
correlation_df = pl.DataFrame(correlation_results)

# You can also create a pivot table to visualize the correlations
correlation_pivot = correlation_df.pivot(
    index="benchmark",
    columns="domain",
    values="rank_correlation"
)

pmi_pivot = correlation_df.pivot(
    index="benchmark",
    columns="domain",
    values="pmi"
)
pmi_matrix = pmi_pivot.drop("benchmark").to_numpy()
correlation_matrix = correlation_pivot.drop("benchmark").to_numpy()

# %%
# Compute the rank correlation between PMI and rank correlation
from scipy.stats import spearmanr

spearmanr(pmi_matrix.flatten(), correlation_matrix.flatten())
# %%
# Compute the rank correlation between PMI and perplexity for each row separately
from scipy.stats import spearmanr

for i in range(pmi_matrix.shape[0]):
    benchmark_name = pmi_pivot.row(i)[0]
    correlation = spearmanr(pmi_matrix[i], correlation_matrix[i])
    print(f"{benchmark_name}: {correlation}")
    
"""
arc_easy: SignificanceResult(statistic=-0.7061814163485556, pvalue=0.0)
piqa: SignificanceResult(statistic=-0.6606237882915195, pvalue=0.0)
sciq: SignificanceResult(statistic=-0.6430416914277367, pvalue=0.0)
lambada_openai: SignificanceResult(statistic=-0.6179755363642303, pvalue=0.0)
lambada_standard: SignificanceResult(statistic=-0.5942536791287508, pvalue=0.0)
lambada_openai_mt_de: SignificanceResult(statistic=0.6983113063710649, pvalue=0.0)
lambada_openai_mt_es: SignificanceResult(statistic=0.6090032217474446, pvalue=0.0)
lambada_openai_mt_fr: SignificanceResult(statistic=0.5420828430744921, pvalue=0.0)
lambada_openai_mt_it: SignificanceResult(statistic=0.6674390991205463, pvalue=0.0)
"""

#%%
# Plot the PMI matrix
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.imshow(pmi_matrix, cmap='viridis', aspect='auto')
plt.colorbar(label='PMI')
plt.xlabel('Domain')
plt.ylabel('Benchmark')
plt.title('PMI Matrix')
# %%
# Plot the correlation matrix
plt.figure(figsize=(10, 4))
plt.imshow(correlation_matrix, cmap='viridis', aspect='auto')
plt.colorbar(label='Correlation')
plt.xlabel('Domain')
plt.ylabel('Benchmark')
plt.title('Correlation Matrix')
# %%
# Plot PMI vs correlation for each benchmark as subplots
fig, axes = plt.subplots(nrows=pmi_matrix.shape[0], ncols=1, figsize=(10, 15))
benchmark_names = pmi_pivot["benchmark"].to_list()

for i in range(pmi_matrix.shape[0]):
    ax = axes[i]
    ax.scatter(pmi_matrix[i], correlation_matrix[i])
    ax.set_xlabel('PMI')
    ax.set_ylabel('Correlation')
    ax.set_title(f'{benchmark_names[i]} - PMI vs Correlation')
    ax.set_xscale('log')
    ax.set_yscale('log')

plt.tight_layout()

# %%
# Compute the ranks for each flattened matrix and scatter plot the ranks

for i in range(pmi_matrix.shape[0]):
    pmi_ranks = np.argsort(np.argsort(pmi_matrix[i]))
    correlation_ranks = np.argsort(np.argsort(correlation_matrix[i]))
    plt.figure(figsize=(10, 8))
    plt.scatter(pmi_ranks, correlation_ranks)
    plt.xlabel('PMI Rank')
    plt.ylabel('Correlation Rank')
# %%
