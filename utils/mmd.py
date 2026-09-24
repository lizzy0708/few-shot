"""
mmd.py — Multi-bandwidth RBF-kernel Maximum Mean Discrepancy (MMD) for domain alignment.

Last attempt (2026-09-24) in the "close domain leakage in gated_nodg z_inv" line of
work, after two prior GRL-based attempts failed (Gram-Schmidt orthogonalization against
domain_classifier's weight subspace, and raising domain_weight to 2.0/5.0 — both either
had no effect on independent-probe domain-acc or actively hurt Acc/F1, see
docs/exec-plans/completed/2026-09-gated-nodg-orth.md and
docs/exec-plans/completed/2026-09-gated-nodg-domain-weight-sweep.md). Both prior attempts
depended on the single GRL domain_classifier discriminator -- a known DANN failure mode
is collapsing to a solution that only fools that one discriminator without actually
aligning the underlying feature distributions. MMD is a discriminator-free alternative:
it minimizes a direct (kernel two-sample-test) distance between per-domain feature
distributions, so it cannot be "fooled" by degenerate encoder solutions the same way.

No new nn.Module / learnable parameters here -- MMD is a plain function of already-
computed pooled features and integer domain labels, called from the training loop.
"""
import torch


def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances between rows of x [n,d] and y [m,d] -> [n,m]."""
    xx = (x * x).sum(dim=1, keepdim=True)          # [n,1]
    yy = (y * y).sum(dim=1, keepdim=True)           # [m,1]
    xy = x @ y.t()                                   # [n,m]
    dist = xx + yy.t() - 2.0 * xy
    return dist.clamp(min=0.0)


def _median_heuristic_sigma(feats: torch.Tensor) -> torch.Tensor:
    """Median-heuristic base bandwidth: sqrt(median of pairwise squared distances / 2),
    computed over ALL samples in the batch (pooled across domains) so the same bandwidth
    set is used for every domain pair in a given forward pass. Falls back to 1.0 if there
    are too few samples (<2) or the median is degenerate (0)."""
    n = feats.size(0)
    if n < 2:
        return torch.tensor(1.0, device=feats.device, dtype=feats.dtype)
    dist = _pairwise_sq_dists(feats, feats)
    iu = torch.triu_indices(n, n, offset=1, device=feats.device)
    off_diag = dist[iu[0], iu[1]]
    if off_diag.numel() == 0:
        return torch.tensor(1.0, device=feats.device, dtype=feats.dtype)
    med = off_diag.median()
    sigma = torch.sqrt(med / 2.0 + 1e-12)
    return torch.where(sigma > 1e-6, sigma, torch.tensor(1.0, device=feats.device, dtype=feats.dtype))


def _multi_rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigmas) -> torch.Tensor:
    """Sum of RBF kernels exp(-dist / (2*sigma^2)) over a list of bandwidths, averaged
    over the number of bandwidths (standard multi-kernel MMD, e.g. Long et al. 2015 DAN)."""
    dist = _pairwise_sq_dists(x, y)
    K = torch.zeros_like(dist)
    for sigma in sigmas:
        K = K + torch.exp(-dist / (2.0 * sigma * sigma + 1e-12))
    return K / len(sigmas)


def _mmd2(x: torch.Tensor, y: torch.Tensor, sigmas) -> torch.Tensor:
    """Squared MMD between two samples x [n,d] and y [m,d] under a multi-bandwidth RBF
    kernel. Uses the U-statistic (diagonal-excluded) estimator for the within-sample
    terms when n (or m) > 1, and falls back to the biased (diagonal-included) mean
    otherwise -- unavoidable with a single sample, and only relevant for very small
    per-domain batch counts."""
    n, m = x.size(0), y.size(0)
    Kxx = _multi_rbf_kernel(x, x, sigmas)
    Kyy = _multi_rbf_kernel(y, y, sigmas)
    Kxy = _multi_rbf_kernel(x, y, sigmas)

    if n > 1:
        kxx = (Kxx.sum() - Kxx.diagonal().sum()) / (n * (n - 1))
    else:
        kxx = Kxx.mean()
    if m > 1:
        kyy = (Kyy.sum() - Kyy.diagonal().sum()) / (m * (m - 1))
    else:
        kyy = Kyy.mean()
    kxy = Kxy.mean()

    return kxx + kyy - 2.0 * kxy


def multi_domain_mmd_loss(feats: torch.Tensor, domain_labels: torch.Tensor,
                           bandwidth_multipliers=(1.0, 2.0, 4.0, 8.0, 16.0),
                           min_samples_per_domain: int = 2) -> torch.Tensor:
    """Average pairwise MMD^2 across all domain pairs present in this batch.

    feats: [B, D] pooled feature (e.g. GAP(z_inv)) -- same space eval-time Mahalanobis
      scoring and the independent domain-invariance probe both use.
    domain_labels: [B] integer domain ids.
    bandwidth_multipliers: median-heuristic base sigma is scaled by each of these and
      the resulting RBF kernels are averaged (multi-kernel MMD, robust to a single bad
      bandwidth choice). Default set matches the "{1,2,4,8,16}x" option discussed for
      this experiment.
    min_samples_per_domain: domains with fewer samples than this in the current batch
      are skipped for numerical stability (small per-domain batch counts are expected
      here since batch_size=16 is split across ~4 calib domains).

    Returns a 0-dim tensor; 0.0 (no grad-bearing terms) if fewer than 2 domains in this
    batch have enough samples to compare.
    """
    device = feats.device
    unique_domains = torch.unique(domain_labels)
    groups = []
    for d in unique_domains:
        mask = domain_labels == d
        if mask.sum().item() >= min_samples_per_domain:
            groups.append(feats[mask])

    if len(groups) < 2:
        return torch.tensor(0.0, device=device, dtype=feats.dtype)

    base_sigma = _median_heuristic_sigma(feats.detach())
    sigmas = [base_sigma * m for m in bandwidth_multipliers]

    total = torch.tensor(0.0, device=device, dtype=feats.dtype)
    n_pairs = 0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            total = total + _mmd2(groups[i], groups[j], sigmas)
            n_pairs += 1

    return total / n_pairs
