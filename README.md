# Transaction Forensics

**Enterprise process analysis with two lenses: CRM pipeline forensics (conformance checking) and NLP communication pattern clustering.**

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://transaction-forensics.vercel.app)
[![Data Source](https://img.shields.io/badge/data-Salesforce%2FHERB-blue)](https://huggingface.co/datasets/Salesforce/HERB)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **Stanford TECH 41** — Building AI Products Through Rapid Prototyping
> Christopher Bailey · March 2026

---

## What It Does

Two analysis tabs in a single viewer:

- **CRM Pipeline Forensics** — Conformance checking of 8,800 Kaggle CRM sales opportunities against an aspirational 8-stage pipeline model. Detects stage skips, reversals, quarter-end compression, velocity anomalies, and account concentration.
- **NLP Communication Patterns** — TF-IDF + KMeans clustering on 37,064 enterprise communications from Salesforce's HERB dataset. Surfaces compliance risks, approval bottlenecks, knowledge silos, and communication gaps.

## Live Demo

**[transaction-forensics.vercel.app](https://transaction-forensics.vercel.app)**

Both tabs are live. CRM tab uses Kaggle CRM Sales Opportunities data; NLP tab uses Salesforce/HERB.

## Pipeline (v3.0)

9 stages, ~109 seconds total (no GPU required):

| # | Stage | Method | Details |
|---|-------|--------|---------|
| 1 | Ingest | File parsing | 37,064 docs from 30 products (Slack, transcripts, docs, PRs) |
| 2 | Normalize | Text cleaning | 31 abbreviation expansions, URL stripping, Slack link unwrapping |
| 3 | BERTopic | SBERT + HDBSCAN | Sentence-transformer embeddings, topic modeling (silhouette: 0.09) |
| 4 | KMeans | TF-IDF + KMeans | 5,000 features, (1,3) n-grams, silhouette-optimized k (silhouette: 0.028) |
| 5 | Network | Communication graph | NetworkX, Louvain communities, centrality analysis, bridge users |
| 6 | Temporal | Change-point detection | Activity windows, volume spikes, temporal clustering |
| 7 | Stability | Bootstrap (50 runs) | Cluster stability via resampled KMeans, pruning unstable clusters |
| 8 | Build cards | Evidence metrics | Severity classification, confidence scoring, pattern typing |
| 9 | Report | JSON + HTML | Pattern cards with provenance, deployed as static viewer |

**Cluster quality caveat:** Global silhouette scores are low (0.028 KMeans / 0.09 BERTopic), indicating weak cluster separation. Findings should be treated as exploratory signals, not confirmed patterns. The Pipeline Transparency section in the viewer documents all parameters.

## Quick Start

```bash
# Clone
git clone https://github.com/chrbailey/transaction-forensics.git
cd transaction-forensics

# Install dependencies
pip install scikit-learn pandas numpy datasets sentence-transformers hdbscan networkx

# Download HERB dataset (one-time, ~200MB)
python -c "from datasets import load_dataset; load_dataset('Salesforce/HERB')"

# Run the full pipeline (~109 seconds)
python3.11 analyze.py

# Open results
open public/index.html
```

## Architecture

```
transaction-forensics/
├── analyze.py              # Main pipeline (9 stages)
├── bertopic_cluster.py     # BERTopic clustering module
├── network_analysis.py     # Communication graph + communities
├── temporal_analysis.py    # Change-point detection
├── public/
│   ├── index.html          # Two-tab viewer (CRM + NLP)
│   └── pattern_cards.json  # NLP analysis output (generated)
└── README.md
```

### Viewer Features

- **Two analysis tabs** — CRM Pipeline Forensics + NLP Communication Patterns
- **Severity filtering** — Critical / High / Medium / Low
- **Type filtering** — Compliance, Bottleneck, Communication, Approval, Anomaly, Escalation
- **Free-text search** — Across titles, descriptions, phrases, products, teams
- **Detail panel** — Evidence quotes, affected scope, recommendations, caveats
- **Pipeline transparency** — Expandable section showing computation stages and parameters
- **Export** — Download findings as JSON
- **Zero dependencies** — No framework, no build step, works offline

## Sample Findings

### CRM Tab (Kaggle Data)

| Finding | Value |
|---------|-------|
| Opportunities analyzed | 8,800 (closed + open) |
| Win rate | 47.2% of closed deals |
| Avg fitness score | 0.05 (against aspirational 8-stage model) |
| Quarter-end concentration | Varies by dataset period |
| Top account concentration (CR5) | Top 5 accounts drive majority of won revenue |

### NLP Tab (HERB Data)

| Severity | Pattern | Documents |
|----------|---------|-----------|
| Critical | Customer data references in public Slack channels | ~12K |
| High | API/integration work siloed across product teams | ~4K |
| Medium | Pull request discussions outside review tools | ~3.5K |
| Medium | Informal approval patterns (LGTM culture) | ~590 |
| Low | Positive sentiment baseline for morale tracking | ~2.2K |

**Note:** NLP cluster confidence scores reflect actual silhouette quality. Low silhouette means these are directional signals, not high-confidence classifications.

## Data Sources

- **CRM tab:** [Kaggle CRM Sales Opportunities](https://www.kaggle.com/datasets/innocentmfa/crm-sales-opportunities) (Apache 2.0 license) — 8,800 opportunities, 85 accounts
- **NLP tab:** [Salesforce/HERB](https://huggingface.co/datasets/Salesforce/HERB) (CC-BY-NC-4.0) — 37,064 enterprise communications across 30 product teams, 120 customers, 18 team members. Note: HERB contains synthetic timestamps extending to 2027.

## Related

The CRM conformance engine, data adapters (SAP RFC, OData, SFDC, CSV, and more), and cross-system correlation logic live in the companion repo: [SAP-Transaction-Forensics](https://github.com/chrbailey/SAP-Transaction-Forensics).

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Analysis | Python 3.11, scikit-learn, pandas, numpy |
| NLP | TF-IDF vectorization, BERTopic (sentence-transformers + HDBSCAN) |
| Network | NetworkX, Louvain community detection |
| Clustering | KMeans + BERTopic with silhouette optimization |
| Frontend | Vanilla HTML/CSS/JS (zero dependencies) |
| Deployment | Vercel (static) |
| Data | HuggingFace Datasets, Kaggle |

## AI Authorship

This project was built with Claude Code (Anthropic). All commits are co-authored as reflected in git history. The architecture, design decisions, and analysis methodology are the author's; the implementation was pair-programmed with AI assistance.

## Author

**Christopher Bailey**
Stanford Continuing Studies — TECH 41: Building AI Products Through Rapid Prototyping
GitHub: [@chrbailey](https://github.com/chrbailey)

## License

MIT
