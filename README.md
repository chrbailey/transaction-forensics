# Transaction Forensics

**Structured data tells you what happened. Unstructured text tells you why.**

Automated process mining engine that combines ERP transaction logs with the unstructured text that surrounds them — emails, Slack messages, help desk tickets, progress reports, and order notes — to surface discrepancies between what organizations report and what actually happened.

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://transaction-forensics.vercel.app)
[![Companion Repo](https://img.shields.io/badge/engine-SAP--Transaction--Forensics-blue)](https://github.com/chrbailey/SAP-Transaction-Forensics)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **Stanford TECH 41** — Building AI Products Through Rapid Prototyping
> Christopher Bailey · March 2026

---

## Live Demo

**[transaction-forensics.vercel.app](https://transaction-forensics.vercel.app)** — 6 analysis tabs, zero dependencies, works offline.

## What It Does

Six forensic lenses on enterprise process data:

| Tab | Data Source | Scale | Key Finding |
|-----|-----------|-------|-------------|
| **Overview** | Architecture + thesis | — | Structured vs. unstructured gap analysis with evidence |
| **CRM Pipeline** | Kaggle CRM Sales Opportunities | 8,800 opportunities | Win rates, velocity patterns, quarter-end compression |
| **BPI Challenge** | BPI Challenge 2019 (4TU.ResearchData) | 251,734 purchase orders, 1.6M events | 57K payment blocks, process variability, resource concentration |
| **IDES Compliance** | SAP IDES Demo System (sap-extractor) | 3,132 cases (O2C + P2P) | **7 compliance violations** in SAP's own reference data |
| **Client Cases** | 3 anonymized consulting engagements | 3M+ ERP records | $103K savings, 2,525 tickets, credit hold overrides, SOD violations |
| **NLP Patterns** | Salesforce/HERB (HuggingFace) | 37,064 documents | 11 communication clusters, approval bottlenecks, knowledge silos |

## The Thesis

Every enterprise system generates two kinds of data:

- **Structured transactions** — timestamps, amounts, stage changes, user IDs — the official story.
- **Unstructured text** — emails, Slack threads, help desk tickets, timesheets, SOWs — what actually happened.

The gap between them is where fraud, waste, and dysfunction hide. This tool automates finding those discrepancies at scale.

**Examples from our analysis:**
- SAP IDES: A Purchase Order was created *before* its Purchase Requisition — approval was documented retroactively. Only detectable by cross-referencing timestamps.
- Salesforce HERB: 1,226 "LGTM/Approved" decisions made in Slack with no audit trail. Formal approval systems show nothing.
- BPI Challenge: 22.7% of purchase orders hit payment blocks, but the event log can't explain why — that answer lives in vendor correspondence.
- Client Case: Sales orders shipped despite "Customer On Credit Hold" flag. The override field tells you it shouldn't have been.

## Real-World Evidence

Three anonymized client engagements demonstrate the pattern:

1. **Healthcare SaaS** — 289 NetSuite users. Automated classification found $103,896/year in waste (dormant accounts, departed employees, replaceable approval-only users). 14.4x ROI, 0.8-month payback.

2. **MedTech Manufacturer (during acquisition)** — 2,525 help desk tickets. Structured ERP data showed "operational." Tickets revealed: dummy transactions as MRP workarounds, data integrity questions ("How did 20413 turn into 20433?"), URGENT escalation culture, and an acquisition wave visible in email domain changes.

3. **Connected Hardware Manufacturer (high-growth)** — 3M+ ERP records (102K sales orders, 97K RMA line items, 43K vendors). Forensic case analysis of 1,090 customer accounts: 28.6% had return events, only 67.5% on-time delivery. ITGC audit (Big Four): 7 users with Administrator role, terminated employee still active, no change management policy. Approval chains so broken that a "reroute approver" field exists in the schema.

## Architecture

The forensic engine (in the [companion repo](https://github.com/chrbailey/SAP-Transaction-Forensics)) processes data through:

```
Data Sources          Adapters (7)         Analysis Engines (4)       Output
─────────────        ──────────────       ─────────────────────      ──────────
SAP ERP (IDES)   →                    →   Conformance Checker    →   Compliance Violations
Salesforce CRM   →   IDataAdapter     →   Temporal Analyzer      →   Bottleneck Reports
BPI Challenge    →   (normalize to    →   Pattern Clustering     →   Pattern Cards
Slack/HERB       →    unified event   →   Cross-System Resolver  →   Evidence Ledger
NetSuite ERP     →    log)
CSV / Custom     →
```

- **834 tests** (602 TypeScript + 232 Python)
- **Deterministic** — all analysis uses `seed=42`
- **Reproducible** — `make demo` for one-command bootstrap

## NLP Pipeline (v3.0)

9 stages, ~109 seconds, no GPU required:

| Stage | Method | Output |
|-------|--------|--------|
| Ingest | File parsing | 37,064 docs from 30 products (32.8K Slack, 3.6K PRs, 400 docs, 321 transcripts) |
| Normalize | Text cleaning | 31 abbreviation expansions, URL/Slack link unwrapping |
| BERTopic | SBERT + HDBSCAN | Sentence-transformer topic modeling (silhouette: 0.09) |
| KMeans | TF-IDF + KMeans | 5,000 features, (1,3) n-grams, silhouette-optimized k (0.028) |
| Network | Communication graph | 521 nodes, 8,406 edges, 7 communities, 5 bridge users |
| Temporal | Change-point detection | Activity windows, volume spikes, temporal clustering |
| Stability | Bootstrap (50 runs) | Cluster stability, pruning unstable patterns |
| Build cards | Evidence metrics | Severity, confidence, pattern typing |
| Report | JSON + HTML | Pattern cards deployed as static viewer |

**Cluster quality caveat:** Low silhouette scores (0.028 KMeans / 0.09 BERTopic) indicate weak cluster separation. Findings are exploratory signals, not confirmed patterns. This is documented transparently in the viewer.

## Quick Start

```bash
git clone https://github.com/chrbailey/transaction-forensics.git
cd transaction-forensics

# Install dependencies
pip install scikit-learn pandas numpy datasets sentence-transformers hdbscan networkx

# Run NLP pipeline (~109 seconds)
python3.11 analyze.py

# View results
open public/index.html
```

For the full forensic engine (conformance checking, data adapters, cross-system correlation):
```bash
git clone https://github.com/chrbailey/SAP-Transaction-Forensics.git
cd SAP-Transaction-Forensics
make demo    # one-command deterministic bootstrap
make test    # 834 tests
```

## Data Sources

| Dataset | License | Records | Used In |
|---------|---------|---------|---------|
| [Kaggle CRM Sales Opportunities](https://www.kaggle.com/datasets/innocentmfa/crm-sales-opportunities) | Apache 2.0 | 8,800 opportunities | CRM tab |
| [BPI Challenge 2019](https://data.4tu.nl/articles/dataset/BPI_Challenge_2019/12715853) | CC BY 4.0 | 251,734 POs, 1.6M events | BPI tab |
| SAP IDES (via [sap-extractor](https://github.com/simonmittag/sap-extractor)) | MIT | 3,132 cases | IDES tab |
| [Salesforce/HERB](https://huggingface.co/datasets/Salesforce/HERB) | CC-BY-NC-4.0 | 37,064 documents | NLP tab |
| Client data (anonymized) | Permission granted | 3M+ records | Client Cases tab |

## How AI Was Used

| What I Did (Christopher Bailey) | What Claude Code Did (AI Pair-Programmer) |
|---|---|
| Defined the problem space and research questions | Implemented data adapters and parsers (TypeScript) |
| Selected data sources and licensed datasets | Built pattern engine and clustering pipeline (Python) |
| Designed the adapter architecture and analysis pipeline | Wrote conformance checking engine |
| Chose conformance algorithms (van der Aalst token replay) | Generated test suites (834 tests) |
| Interpreted findings and wrote forensic narratives | Built the 6-tab dashboard (vanilla HTML/CSS/JS) |
| Determined what's a real finding vs. a statistical artifact | Statistical computations (effect sizing, CI, p-values) |
| Real-world client engagements and domain expertise (20 yrs ERP) | All code in git history with co-author tags |

Claude Code is a force multiplier. The AI doesn't know what's worth finding — it doesn't know that a PO-before-PR is a Sarbanes-Oxley risk, or that 22.7% payment block rates are 4x industry norms. Domain expertise decides what to look for; AI makes looking fast.

## Author

**Christopher Bailey**
Stanford Continuing Studies — TECH 41: Building AI Products Through Rapid Prototyping
GitHub: [@chrbailey](https://github.com/chrbailey)

## License

MIT
