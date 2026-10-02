# Metrics

Let the centered clean RGB patch covariance be `Sigma = V diag(lambda) V^T`, the raw patch embedding be `W`, and `G = W^T W`.

- **Absolute directional gain:** `g_i = v_i^T G v_i = ||W v_i||_2^2`. The rankwise CSV retains every `lambda_i` and `g_i`.
- **Mean-normalized rankwise gain:** `g_i / ((1/768) sum_j g_j)`. The clean-patch baseline uses the matching `lambda_i / mean_j(lambda_j)`. These are the quantities in the log-scale rankwise plot.
- **Top-k cumulative gain:** `sum_{i<=k} g_i / sum_i g_i`. The clean baseline is cumulative explained variance `sum_{i<=k} lambda_i / sum_i lambda_i`.
- **Rank-bin gain share:** the bar chart reports gain mass in `1:64`, `1:128`, `1:256`, `513:768`, `641:768`, and `705:768`. These intentionally overlapping bins summarize progressively wider head and tail regions.
- **Noisy raw-patch spectrum:** with the paper convention `x_t=t x_0+(1-t) epsilon` and unit isotropic Gaussian `epsilon`, direction `i` has eigenvalue `t^2 lambda_i+(1-t)^2`. The spectrum and cumulative-spectrum figures show `t in {1.00,0.75,0.50,0.25,0.00}` on a linear rank axis; `t=1` is clean and `t=0` is pure noise.
- **Embedding energy:** `||W||_F`; equivalently `tr(G) = sum_i g_i`. Both absolute columns are retained.
- **Top-k gain share:** `sum_{i<=k} g_i / sum_i g_i`; **bottom-100 share** uses the final 100 raw-PCA directions.
- **Lambda amplification:** `(sum_i lambda_i g_i / sum_i g_i) / (sum_i lambda_i / 768)`.
- **Gain effective rank:** `exp(-sum_i p_i log p_i)`, `p_i=g_i/sum_j g_j`.
- **Gram effective/stable rank:** entropy rank of eigenvalues of `G`, and `tr(G)/lambda_max(G)`.
- **Log-spectrum Spearman:** rank correlation between `log(lambda_i)` and `log(g_i)`; it tests ordering, not magnitude.
- **Gram/covariance cosine:** `<G,Sigma>_F/(||G||_F ||Sigma||_F)`.
- **PCA off-diagonal ratio:** `||offdiag(V^T G V)||_F / ||V^T G V||_F`.
- **Top-k principal alignment:** `||V_k^T U_k||_F^2/k`, where `U_k` is the leading eigenspace of `G`. Random expectation is `k/768`.
- **Log alignment heatmap:** the directional heatmaps display `log10(|v_i^T u_j|^2 + 1e-8)`. Log scaling changes visibility only; CSV values remain linear.

All reported matrices are raw checkpoint states, not EMA. HyperDiT-B uses the P16 semantic stem `large_embedder.proj.weight`; its P8 fine stem is excluded.
