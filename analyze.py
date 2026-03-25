#!/usr/bin/env python3
"""
Transaction Forensics — Pattern Engine
Analyzes Salesforce HERB enterprise communication data using NLP clustering.

Pipeline: Ingest → Normalize → TF-IDF Vectorize → KMeans Cluster → Correlate → Report

Author: Christopher Bailey
Data Source: Salesforce/HERB (HuggingFace)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

HERB_BASE = os.environ.get(
    "HERB_PATH",
    "/Volumes/OWC drive/Models/huggingface-cache/hub/datasets--Salesforce--HERB/"
    "snapshots/a00bca08f9118e482e6de9951fdcb654fbed5343"
)

OUTPUT_DIR = Path(__file__).parent / "public"
N_CLUSTERS = 12
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 3)
MIN_DOC_LENGTH = 30

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: INGEST
# ═══════════════════════════════════════════════════════════════════════════

def ingest_herb(base_path: str) -> dict:
    """Load all HERB data: Slack, transcripts, documents, PRs, metadata."""
    t0 = time.time()
    base = Path(base_path)

    # Metadata
    with open(base / "metadata" / "customers_data.json") as f:
        customers = json.load(f)
    with open(base / "metadata" / "salesforce_team.json") as f:
        team = json.load(f)

    # Products
    products = {}
    for p in sorted((base / "products").iterdir()):
        if p.suffix == ".json":
            with open(p) as f:
                products[p.stem] = json.load(f)

    # Extract documents
    documents = []
    for prod_name, prod_data in products.items():
        # Slack messages
        for msg in prod_data.get("slack", []):
            user = msg.get("Message", {}).get("User", {})
            text = user.get("text", "")
            if text and len(text) >= MIN_DOC_LENGTH and "created this channel" not in text:
                documents.append({
                    "text": text,
                    "source": "slack",
                    "product": prod_name,
                    "channel": msg.get("Channel", {}).get("name", ""),
                    "user": user.get("userId", ""),
                    "timestamp": user.get("timestamp", ""),
                })

        # Meeting transcripts
        for t_item in prod_data.get("meeting_transcripts", []):
            transcript = t_item.get("transcript", "")
            if transcript and len(transcript) >= MIN_DOC_LENGTH:
                documents.append({
                    "text": transcript[:2000],  # Truncate long transcripts
                    "source": "transcript",
                    "product": prod_name,
                    "channel": "",
                    "user": "",
                    "timestamp": "",
                })

        # Documents
        for doc in prod_data.get("documents", []):
            content = doc.get("content", "")
            if content and len(content) >= MIN_DOC_LENGTH:
                documents.append({
                    "text": content[:2000],
                    "source": "document",
                    "product": prod_name,
                    "channel": "",
                    "user": "",
                    "timestamp": "",
                })

        # PR descriptions
        for pr in prod_data.get("prs", []):
            title = pr.get("title", "")
            summary = pr.get("summary", "")
            combined = f"{title}. {summary}" if summary else title
            if combined and len(combined) >= MIN_DOC_LENGTH:
                documents.append({
                    "text": combined[:500],
                    "source": "pull_request",
                    "product": prod_name,
                    "channel": "",
                    "user": "",
                    "timestamp": "",
                })

    ingest_time = time.time() - t0
    return {
        "documents": documents,
        "customers": customers,
        "team": team,
        "products": list(products.keys()),
        "stats": {
            "total_documents": len(documents),
            "by_source": dict(Counter(d["source"] for d in documents)),
            "by_product": dict(Counter(d["product"] for d in documents)),
            "ingest_time_seconds": round(ingest_time, 3),
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: NORMALIZE
# ═══════════════════════════════════════════════════════════════════════════

# Salesforce/enterprise abbreviations to expand
SF_ABBREVIATIONS = {
    "SFDC": "Salesforce",
    "SOQL": "Salesforce Object Query Language",
    "SOSL": "Salesforce Object Search Language",
    "DML": "data manipulation language",
    "VF": "Visualforce",
    "LWC": "Lightning Web Component",
    "SLDS": "Salesforce Lightning Design System",
    "CPQ": "configure price quote",
    "MQL": "marketing qualified lead",
    "SQL": "sales qualified lead",
    "ACV": "annual contract value",
    "ARR": "annual recurring revenue",
    "TAM": "total addressable market",
    "NPS": "net promoter score",
    "CSAT": "customer satisfaction",
    "SLA": "service level agreement",
    "PII": "personally identifiable information",
    "GDPR": "General Data Protection Regulation",
    "SOC": "service organization control",
    "RBAC": "role-based access control",
    "CI/CD": "continuous integration continuous deployment",
    "PR": "pull request",
    "NLP": "natural language processing",
    "API": "application programming interface",
    "SDK": "software development kit",
    "MVP": "minimum viable product",
    "POC": "proof of concept",
    "KPI": "key performance indicator",
    "OKR": "objectives and key results",
    "SWOT": "strengths weaknesses opportunities threats",
    "ROI": "return on investment",
}

# Boilerplate patterns to strip
BOILERPLATE_PATTERNS = [
    r"https?://\S+",  # URLs
    r"@eid_[a-f0-9]+",  # Employee ID mentions (keep for audit pattern)
    r"<[^>]+\|([^>]+)>",  # Slack link formatting → keep text
]

def normalize_text(text: str) -> str:
    """Clean and normalize text for clustering."""
    # Expand Slack links: <url|text> → text
    text = re.sub(r"<[^>]+\|([^>]+)>", r"\1", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Expand abbreviations
    for abbr, expansion in SF_ABBREVIATIONS.items():
        text = re.sub(rf"\b{re.escape(abbr)}\b", f"{abbr} ({expansion})", text, count=1)
    # Lowercase
    text = text.lower().strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3: VECTORIZE + CLUSTER
# ═══════════════════════════════════════════════════════════════════════════

def cluster_documents(documents: list[dict], n_clusters: int = N_CLUSTERS) -> dict:
    """TF-IDF vectorization + KMeans clustering."""
    t0 = time.time()

    # Normalize texts
    texts = [normalize_text(d["text"]) for d in documents]

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words="english",
        min_df=2,
        max_df=0.95,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Optimal k estimation via silhouette
    k_range = range(max(2, n_clusters - 4), n_clusters + 5)
    best_k, best_score = n_clusters, -1
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, labels, sample_size=min(5000, len(texts)))
        if score > best_score:
            best_k, best_score = k, score

    # Final clustering with best k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    # Extract top phrases per cluster
    cluster_phrases = {}
    cluster_docs = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        cluster_docs[label].append(i)

    for label, doc_indices in cluster_docs.items():
        # Mean TF-IDF scores for this cluster
        cluster_vectors = tfidf_matrix[doc_indices].toarray()
        mean_scores = cluster_vectors.mean(axis=0)
        top_indices = mean_scores.argsort()[-10:][::-1]
        phrases = [feature_names[idx] for idx in top_indices if mean_scores[idx] > 0]
        cluster_phrases[label] = phrases

    cluster_time = time.time() - t0

    return {
        "labels": cluster_labels.tolist(),
        "n_clusters": best_k,
        "silhouette_score": round(best_score, 4),
        "cluster_sizes": dict(Counter(cluster_labels.tolist())),
        "cluster_phrases": {str(k): v for k, v in cluster_phrases.items()},
        "cluster_docs": {str(k): v for k, v in cluster_docs.items()},
        "tfidf_features": len(feature_names),
        "tfidf_ngram_range": list(TFIDF_NGRAM_RANGE),
        "cluster_time_seconds": round(cluster_time, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4: CORRELATE — analyze clusters for patterns
# ═══════════════════════════════════════════════════════════════════════════

SEVERITY_RULES = {
    "compliance": {"keywords": ["privacy", "gdpr", "pii", "credentials", "security", "data handling", "sensitive"], "default": "critical"},
    "bottleneck": {"keywords": ["approval", "waiting", "blocked", "sign-off", "escalate", "delay"], "default": "high"},
    "escalation": {"keywords": ["frustrated", "third time", "no response", "urgent", "critical"], "default": "high"},
    "communication": {"keywords": ["silo", "isolated", "no overlap", "timezone", "async"], "default": "medium"},
    "anomaly": {"keywords": ["unusual", "imbalance", "concentration", "outlier", "deviation"], "default": "medium"},
    "approval": {"keywords": ["swot", "risk", "mitigation", "unresolved", "unaddressed"], "default": "medium"},
}

def classify_pattern(phrases: list[str], docs: list[dict]) -> tuple[str, str]:
    """Classify pattern type and severity from cluster content."""
    text_blob = " ".join(phrases + [d.get("text", "")[:200] for d in docs[:10]]).lower()

    for ptype, rules in SEVERITY_RULES.items():
        hits = sum(1 for kw in rules["keywords"] if kw in text_blob)
        if hits >= 2:
            return ptype, rules["default"]

    return "anomaly", "medium"

def correlate_clusters(documents: list[dict], clustering: dict) -> list[dict]:
    """Analyze each cluster and generate pattern cards."""
    t0 = time.time()
    cards = []

    for label_str, doc_indices in clustering["cluster_docs"].items():
        label = int(label_str)
        if len(doc_indices) < 3:  # Skip tiny clusters
            continue

        cluster_docs = [documents[i] for i in doc_indices]
        phrases = clustering["cluster_phrases"].get(label_str, [])

        # Classify
        ptype, severity = classify_pattern(phrases, cluster_docs)

        # Source breakdown
        sources = Counter(d["source"] for d in cluster_docs)
        products = Counter(d["product"] for d in cluster_docs)
        channels = Counter(d["channel"] for d in cluster_docs if d["channel"])

        # Time range
        timestamps = [d["timestamp"] for d in cluster_docs if d["timestamp"]]
        time_range = ""
        if timestamps:
            ts_sorted = sorted(timestamps)
            time_range = f"{ts_sorted[0][:10]} to {ts_sorted[-1][:10]}"

        # Sample messages (prefer slack messages for readability)
        samples = [d["text"][:200] for d in cluster_docs if d["source"] == "slack"][:3]
        if not samples:
            samples = [d["text"][:200] for d in cluster_docs[:3]]

        # Generate title from top phrases
        title_phrases = [p for p in phrases[:3] if len(p) > 3]
        title = " / ".join(w.title() for w in title_phrases[:3]) if title_phrases else f"Cluster {label} Pattern"

        # Confidence based on cluster cohesion (size + phrase strength)
        size_factor = min(len(doc_indices) / 50, 1.0)
        conf_score = round(0.5 + (size_factor * 0.4) + (0.1 if len(phrases) > 5 else 0), 2)
        conf_score = min(conf_score, 0.95)
        confidence = "HIGH" if conf_score >= 0.8 else "MEDIUM" if conf_score >= 0.6 else "LOW"

        card = {
            "id": f"PAT-{hash(title) % 0xFFFFFF:06X}",
            "title": title,
            "description": f"Cluster of {len(doc_indices)} documents sharing phrases: {', '.join(phrases[:5])}. "
                          f"Found across {len(products)} products and {len(sources)} source types.",
            "type": ptype,
            "confidence": confidence,
            "confidence_score": conf_score,
            "severity": severity,
            "top_phrases": phrases[:6],
            "occurrence": f"Found in {len(doc_indices)} of {len(documents)} documents",
            "effect": f"Pattern spans {len(products)} product teams with {sources.get('slack', 0)} messages, "
                     f"{sources.get('transcript', 0)} transcripts, {sources.get('document', 0)} docs.",
            "evidence": {
                "source_count": len(doc_indices),
                "source_breakdown": dict(sources),
                "sample_messages": samples,
                "affected_products": list(products.keys()),
                "affected_teams": list(channels.keys())[:5] if channels else list(products.keys()),
                "time_range": time_range or "Not timestamped",
            },
            "sample_snippets": [s[:100] for s in samples[:2]],
            "caveats": f"Cluster confidence: {conf_score:.0%}. Based on TF-IDF similarity — verify semantic relevance.",
            "recommendation": f"Review the {len(doc_indices)} documents in this cluster for actionable patterns. "
                            f"Top signal phrases: {', '.join(phrases[:3])}.",
            "computation": {
                "cluster_id": label,
                "cluster_size": len(doc_indices),
                "method": "TF-IDF + KMeans",
                "top_tfidf_terms": phrases[:6],
            }
        }
        cards.append(card)

    # Sort by severity then confidence
    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    cards.sort(key=lambda c: (sev_order.get(c["severity"], 9), -c["confidence_score"]))

    correlate_time = time.time() - t0
    return cards, round(correlate_time, 3)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 5: REPORT
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(ingest_result: dict, clustering: dict, cards: list[dict],
                    correlate_time: float) -> dict:
    """Package everything into the final output."""
    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "author": "Christopher Bailey",
            "engine": "Transaction Forensics Pattern Engine v1.0",
            "source": "Salesforce/HERB",
            "source_url": "https://huggingface.co/datasets/Salesforce/HERB",
            "license": "CC-BY-NC-4.0",
            "products_analyzed": ingest_result["products"],
            "total_messages_scanned": ingest_result["stats"]["by_source"].get("slack", 0),
            "total_documents_scanned": ingest_result["stats"]["by_source"].get("document", 0),
            "total_transcripts_scanned": ingest_result["stats"]["by_source"].get("transcript", 0),
            "total_prs_analyzed": ingest_result["stats"]["by_source"].get("pull_request", 0),
            "total_customers": len(ingest_result["customers"]),
            "total_team_members": len(ingest_result["team"]),
        },
        "pipeline": {
            "stages": [
                {"name": "Ingest", "duration_seconds": ingest_result["stats"]["ingest_time_seconds"],
                 "documents_loaded": ingest_result["stats"]["total_documents"],
                 "sources": ingest_result["stats"]["by_source"]},
                {"name": "Normalize", "description": f"Text cleaning + {len(SF_ABBREVIATIONS)} abbreviation expansions + URL/mention stripping"},
                {"name": "Vectorize", "method": "TF-IDF",
                 "features": clustering["tfidf_features"],
                 "ngram_range": clustering["tfidf_ngram_range"],
                 "max_features": TFIDF_MAX_FEATURES},
                {"name": "Cluster", "method": "KMeans",
                 "optimal_k": clustering["n_clusters"],
                 "silhouette_score": clustering["silhouette_score"],
                 "duration_seconds": clustering["cluster_time_seconds"],
                 "cluster_sizes": clustering["cluster_sizes"]},
                {"name": "Correlate", "patterns_found": len(cards),
                 "duration_seconds": correlate_time,
                 "severity_distribution": dict(Counter(c["severity"] for c in cards))},
            ],
            "total_duration_seconds": round(
                ingest_result["stats"]["ingest_time_seconds"] +
                clustering["cluster_time_seconds"] + correlate_time, 3
            ),
        },
        "cards": cards,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Transaction Forensics — Pattern Engine v1.0")
    print(f"Author: Christopher Bailey")
    print(f"Data: Salesforce/HERB (HuggingFace)")
    print("=" * 60)

    # Stage 1: Ingest
    print("\n[1/5] Ingesting HERB dataset...")
    ingest_result = ingest_herb(HERB_BASE)
    stats = ingest_result["stats"]
    print(f"      Loaded {stats['total_documents']} documents in {stats['ingest_time_seconds']}s")
    print(f"      Sources: {stats['by_source']}")
    print(f"      Products: {len(ingest_result['products'])}")

    # Stage 2: Normalize (happens inside vectorize)
    print(f"\n[2/5] Normalizing text ({len(SF_ABBREVIATIONS)} abbreviation expansions)...")

    # Stage 3: Vectorize + Cluster
    print(f"\n[3/5] TF-IDF vectorization + KMeans clustering...")
    clustering = cluster_documents(ingest_result["documents"])
    print(f"      Features: {clustering['tfidf_features']}")
    print(f"      Optimal k: {clustering['n_clusters']} (silhouette: {clustering['silhouette_score']})")
    print(f"      Cluster sizes: {clustering['cluster_sizes']}")
    print(f"      Time: {clustering['cluster_time_seconds']}s")

    # Stage 4: Correlate
    print(f"\n[4/5] Correlating clusters → pattern cards...")
    cards, correlate_time = correlate_clusters(ingest_result["documents"], clustering)
    print(f"      Generated {len(cards)} pattern cards in {correlate_time}s")
    for c in cards:
        print(f"      {c['severity']:8} | {c['type']:15} | {c['title'][:50]}")

    # Stage 5: Report
    print(f"\n[5/5] Generating report...")
    report = generate_report(ingest_result, clustering, cards, correlate_time)

    # Write output
    output_path = OUTPUT_DIR / "pattern_cards.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n      Output: {output_path}")
    print(f"      Total pipeline time: {report['pipeline']['total_duration_seconds']}s")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
