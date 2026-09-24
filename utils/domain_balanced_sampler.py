"""
domain_balanced_sampler.py — 2026-09-24 MMD batch-balancing attempt.

Motivation: mmd_weight sweep {10.0, 1.0, 0.1} with plain shuffle=True DataLoader all
failed the pre-registered success criteria (docs/exec-plans/completed/2026-09-gated-nodg-mmd.md),
showing a monotonic Acc/F1-damage-vs-domain-acc-reduction trade-off with no sweet spot.
Suspected root cause (not yet isolated): batch_size=16 split randomly across the 4 calib
train domains gives each domain only ~4 samples/batch on average with high variance
(sometimes 0-1 samples from a domain in a batch) -- the multi-bandwidth RBF-kernel MMD
estimate in utils/mmd.py is a *biased* estimator whose bias and variance both blow up as
per-domain sample count shrinks, so the "MMD loss" being minimized may have been mostly
noise rather than a meaningful alignment signal. This module isolates that variable by
making every batch exactly domain-balanced, so it can be tested independently of
mmd_weight itself (see docs/exec-plans/active/2026-09-mmd-balanced-batch.md).

Does NOT change utils/mmd.py or the loss formula -- only how batches are assembled before
being handed to the model.
"""
import random
import torch
from torch.utils.data import Sampler


class DomainBalancedBatchSampler(Sampler):
    """
    Yields batches where each of the `num_domains` domains (assumed to correspond,
    in order, to the `num_domains` sub-datasets of a torch ConcatDataset with
    contiguous index ranges -- i.e. train_gated_mask_model.py's
    `ConcatDataset([HUSTDataset(domain=d) for d in train_domains])`) contributes
    exactly `batch_size // num_domains` samples to every batch.

    Contrast with the previous behavior (shuffle=True on the flat ConcatDataset):
    that gives each domain only batch_size/num_domains samples *in expectation*,
    with substantial per-batch variance (binomial sampling without domain
    stratification). This sampler removes that variance entirely.

    Each domain's per-epoch pool is independently shuffled and consumed without
    replacement; epoch length = floor(min_domain_size / per_domain) batches, so a
    few samples from larger domains may be dropped each epoch (acceptable -- normal
    window counts are already close to balanced across the 4 calib domains, ~1497
    each per the 2026-09-24 preprocessing re-verification).
    """

    def __init__(self, concat_dataset, batch_size, num_domains, seed=0):
        if batch_size % num_domains != 0:
            raise ValueError(
                f"batch_size={batch_size} must be divisible by num_domains={num_domains} "
                f"for exact per-batch domain balance"
            )
        self.per_domain = batch_size // num_domains
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        cum = list(concat_dataset.cumulative_sizes)
        starts = [0] + cum[:-1]
        ends = cum
        self.domain_ranges = list(zip(starts, ends))
        assert len(self.domain_ranges) == num_domains, (
            f"ConcatDataset has {len(self.domain_ranges)} sub-datasets, expected "
            f"num_domains={num_domains} -- one HUSTDataset per train domain"
        )

        sizes = [e - s for s, e in self.domain_ranges]
        self.num_batches = min(sizes) // self.per_domain
        if self.num_batches == 0:
            raise ValueError(
                f"Smallest domain has only {min(sizes)} samples, need at least "
                f"per_domain={self.per_domain} (batch_size={batch_size} / "
                f"num_domains={num_domains})"
            )

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        shuffle_seed = self.seed * 100000 + self.epoch
        self.epoch += 1

        per_domain_shuffled = []
        for s, e in self.domain_ranges:
            idx = (torch.randperm(e - s, generator=g) + s).tolist()
            per_domain_shuffled.append(idx)

        for b in range(self.num_batches):
            batch = []
            for idx_list in per_domain_shuffled:
                batch.extend(idx_list[b * self.per_domain:(b + 1) * self.per_domain])
            random.Random(shuffle_seed * 1000 + b).shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches
