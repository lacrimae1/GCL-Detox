# GCL-Detox

**Disentangling Uniformity for Robust and Effective Graph Contrastive Learning-Based Recommendation**

Qinwen Ran, Guanmeng Xian, Ning Yang (Sichuan University), Philip S. Yu (University of Illinois Chicago)

---

## Overview

Graph Contrastive Learning (GCL) improves recommendation by pushing embeddings toward **uniformity** on the unit hypersphere — but that same geometric property opens an attack surface. Poisoning attacks such as CLeaR deliberately amplify uniformity to promote unpopular target items, and existing defenses, which focus on filtering suspicious users or generic adversarial training, leave this weakness untouched.

**GCL-Detox** is a theory-grounded defense that treats embedding geometry itself as the thing to protect. Instead of jointly optimizing the contrastive and ranking losses (where their gradients keep interfering), it **decouples them in time** through a tri-phase schedule with progressive annealing:

1. **Contrastive Pretraining** — optimize the contrastive loss alone to build a semantically meaningful embedding space.
2. **Progressive Transition** — linearly anneal the contrastive loss while ramping up the BPR loss, letting the geometry migrate smoothly from uniformity-dominated to clustering-oriented.
3. **BPR Fine-tuning** — optimize the ranking objective only, sharpening accuracy while shrinking the uniformity-exploitable attack surface.


## Experimental Setup

| | |
| --- | --- |
| Datasets | ML-10M, DouBan, Yelp |
| Backbone | LightGCN |
| GCL objectives defended | SGL, SimGCL, XSimGCL |
| Attacks | Rand, PGA, A_ra, CLeaR (+ NoAtt) |
| Baseline defenses | Degree-SAD, Anti-FU, Trust-GRS |
| Metrics | NDCG@50 (utility), HR@50 on target items (robustness), Trade_off |

Split 7:1:2, five target items sampled from the bottom 20% of the popularity ranking, three runs averaged.

## Citation

```bibtex
@inproceedings{ran2026gcldetox,
  author    = {Ran, Qinwen and Xian, Guanmeng and Yang, Ning and Yu, Philip S.},
  title     = {GCL-Detox: Disentangling Uniformity for Robust and Effective Graph Contrastive Learning-Based Recommendation},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26)},
  year      = {2026},
  address   = {Rome, Italy},
  publisher = {ACM},
  doi       = {10.1145/3799682.3840838}
}
```
