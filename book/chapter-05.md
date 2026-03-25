# Chapter 5 - Graph Thinking and Record Linkage

> Not all data fits in rows and columns. Relationships between entities are data too, and ignoring them means ignoring information that can improve data quality and analysis.

> Full runnable code for all examples is in `examples/chapter-05/`.

```{admonition} Who is this for?
If you have completed Chapters 1-4 (regression/classification, cross-validation, decision trees, and neural networks), you are ready. This chapter introduces a different way of thinking about data: networks of connected entities rather than independent rows.
```

## Learning goals

- Identify when a dataset has graph structure.
- Create and visualize graphs with networkx using real survey data concepts.
- Compute basic graph metrics (degree, components, clustering coefficient) and interpret them for data quality.
- Explain the record linkage problem and distinguish deterministic from probabilistic matching.
- Describe a complete linkage pipeline: blocking, comparison, scoring, clustering.
- Evaluate a linked dataset: what questions to ask before using it for analysis.
- Describe how graph structure affects survey estimates (household effects, geographic spillover).

```{admonition} Why this matters for federal statistics
:class: tip
Federal statistical agencies link records across dozens of administrative and survey data sources. Every American Community Survey response is linked to administrative records. Every business in the economic censuses is tracked across time through an identifier system. Vital statistics are linked to hospital records. Understanding graph structure and record linkage lets you:

- Evaluate the quality of linked datasets before using them for analysis.
- Understand error propagation when linkage goes wrong (false matches create spurious relationships; missed matches create artificial coverage gaps).
- Ask the right questions when a data integration project is proposed: what is the match rate? What is the false positive rate? How were ambiguous links resolved?
- Design data collection that facilitates future linkage (consistent identifier formats, field validation).
```

---

## 1. Setup

This chapter uses networkx for graph operations, pandas for data manipulation, and scikit-learn for the linkage classifier. All code is in `examples/chapter-05/`.

```{code-block} python
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from sklearn.linear_model import LogisticRegression
```

---

## 2. When is your data a graph?

A graph has *nodes* (entities) and *edges* (relationships between entities). Many datasets in federal statistics have graph structure that standard rectangular data formats obscure.

| Data source | Nodes | Edges | Example edge meaning |
|-------------|-------|-------|----------------------|
| ACS household roster | Persons | Household membership | "lives with" |
| Employer-employee records | Persons + Establishments | Employment | "works at" |
| Geographic hierarchy | States, Counties, Tracts | Containment | "is part of" |
| Administrative records | Person records | Identity match | "is the same person as" |
| Trade flow data | Countries / States | Transaction | "exports to" |
| Citation networks | Papers / Authors | Citation | "cites" |

The three graph types most relevant to federal statistics appear in `examples/chapter-05/02_graph_types_visualization.py`:

- *Household network*: undirected, with relationship-typed edges (spouse, parent-child, neighbor). Clustering within households is the source of the design effects discussed in Section 6.
- *Geographic containment hierarchy*: directed acyclic graph (DAG) from state down to census tract. Data aggregated along this hierarchy must respect the directed structure.
- *Employer-employee bipartite graph*: two node types (firms, workers) with employment edges. Worker 2 appears at both Firm A and Firm B — a fact invisible in a single rectangular table.

---

## 3. Graph basics with networkx

### 3.1 Building a household network

A Census block contains multiple households. Within each household, people are connected by family relationships. Across households, connections arise from neighbor ties, co-worker relationships, or shared employer records.

The code in `examples/chapter-05/01_household_network.py` builds this network for a synthetic Census block of ten people across five households.

```{code-block} python
G = nx.Graph()

# Nodes: people with demographic attributes
G.add_node("H001-P01", hh_id="H001", age=43, income=65000)

# Edges: within-household (strong) and cross-household (weak)
G.add_edge("H001-P01", "H001-P02", rel="spouse", weight=3)
G.add_edge("H001-P01", "H003-P01", rel="neighbor", weight=1)
```

The edge `weight` attribute encodes relationship strength. Within-household edges (weight 3) are darker in the visualization than neighbor ties (weight 1). This distinction matters when computing graph metrics for data quality assessment.

### 3.2 Basic graph metrics

Three metrics are particularly useful when evaluating linked datasets:

*Degree*: the number of connections a node has. In a linkage graph, a person-node with unusually high degree may represent a real hub (a householder with many dependents) or a quality problem (multiple records incorrectly merged into one). Both situations look the same in the degree distribution — distinguishing them requires manual review of the flagged records.

*Connected components*: groups of nodes that are reachable from each other. In a household network, each component is typically one household. An isolated node (degree 0) may be a genuine single-person household, or it may indicate a linkage failure where this person's records appear in another dataset but were not matched.

*Clustering coefficient*: the fraction of a node's neighbors who are also connected to each other. In a household, all members are usually connected to all other members, so the clustering coefficient is high. This is what creates intraclass correlation, discussed in Section 6.

```{code-block} python
degrees    = dict(G.degree())
components = list(nx.connected_components(G))
cc         = nx.average_clustering(G.subgraph(household_nodes))

print(f"Components: {len(components)}")
print(f"Average clustering coefficient: {cc:.3f}")
```

---

## 4. The record linkage problem

### 4.1 What is record linkage?

Record linkage is the process of identifying records in one or more datasets that refer to the same real-world entity (Newcombe, Kennedy, Axford & James, 1959). Also called *entity resolution*, *data deduplication*, or *data integration*.

The challenge: entities do not come with unique identifiers across all data sources. A person appears in the ACS as "Maria Garcia, DOB 1985-04-12, 123 Main St, Austin TX" and in tax records as "M. Garcia, 4/12/1985, 123 Main Street, Austin, Texas." These are the same person, but no exact match exists. The linkage algorithm must decide how similar is similar enough. For a comprehensive treatment of record linkage methodology, see Christen (2012).

### 4.2 Deterministic vs. probabilistic matching

*Deterministic matching* links records that agree exactly on one or more blocking keys:
- Exact SSN match
- Exact name + DOB + ZIP combination
- Assigned identifier (EIN for businesses, state ID for drivers)

Simple and fast, but misses links where any field has errors, abbreviations, or missing values. Most administrative data has enough quality variation that deterministic matching alone leaves substantial coverage gaps.

*Probabilistic matching* (Fellegi & Sunter, 1969) computes a score for every candidate pair based on partial agreement across multiple fields:
- Name similarity: 0.85 (not exact, but close)
- DOB exact match: 1.0
- Address similarity: 0.72
- Combined score: classified as match or non-match at a threshold

The threshold is a design choice with real consequences, discussed in the blocking trade-off section below.

Three comparison functions are used throughout this chapter (full implementations in `examples/chapter-05/05_comparison_scoring.py`):

- `string_similarity(a, b)`: SequenceMatcher ratio, 0.0 to 1.0. NaN-safe.
- `dob_match(a, b)`: returns 1.0 for exact DOB, 0.5 if year-only matches (handles month/day transpositions), 0.0 otherwise.
- `address_similarity(a, b)`: alias for string similarity on address fields.

### 4.3 Why record linkage is a graph problem

After computing match scores, we have a *bipartite candidate graph*: records from Source A on one side, records from Source B on the other, and edges representing "possible match" pairs above a threshold.

When we accept a set of edges as matches, the resulting linked records form *clusters* (connected components). A single person should form a cluster of size 2 (one record from each source). The graph structure makes the error modes visible:

- Clusters of size 1: unlinked singletons. This person was not matched. May be correct (not in both sources) or a missed match.
- Clusters of size 2: the ideal case. One record from each source, linked with high confidence.
- Clusters of size 3+: multi-record clusters. Either a true many-to-one relationship (one person with records in multiple systems) or a false match chain that linked distinct people together.

The bipartite visualization in `examples/chapter-05/02_graph_types_visualization.py` shows accepted matches (blue edges) alongside rejected pairs (red edges) so the threshold's effect is visible geometrically.

---

## 5. End-to-end record linkage pipeline

The pipeline has four stages:

```
Source A + Source B
       |
   [BLOCKING]        Reduce 40,000 pairs to ~1,200 candidates
       |
  [COMPARISON]       Compute name_sim, dob_sim, addr_sim for each pair
       |
   [SCORING]         Logistic regression: P(match | features)
       |
  [CLUSTERING]       Connected components of accepted match graph
       |
   Linked file + cluster size distribution
```

### 5.1 Synthetic records with noise

`examples/chapter-05/03_synthetic_records.py` generates 200 true entities who appear in two datasets: a survey (Source A, clean) and an administrative file (Source B, noisy). Source B contains records for only 180 of the 200 people, plus 20 new entities not in Source A.

This setup mirrors operational reality: no administrative file is a perfect superset of a survey roster, and some survey respondents are absent from the admin data entirely.

The noise functions model real-world data quality problems:

| Noise type | Example | Cause in real data |
|------------|---------|-------------------|
| Name abbreviation | "Maria Garcia" -> "M. Garcia" | Different data entry conventions |
| Name typo | "Garcia" -> "Garsia" | Transcription error |
| DOB transposition | "1985-04-12" -> "1985-12-04" | Month/day swap |
| Address abbreviation | "Main Street" -> "Main St" | No standardization requirement |
| Missing field | NaN | Field not collected in one source |

After running `03_synthetic_records.py`, two files are written: `source_a.csv` and `source_b.csv`.

### 5.2 Blocking: reducing the comparison space

Without blocking, comparing all 200 Source A records against all 200 Source B records requires 200 × 200 = 40,000 comparisons. For a realistic dataset of 10 million records per source, that is 10^14 comparisons — computationally impossible.

Blocking restricts comparisons to pairs that share a coarse key. Two blocking keys are used here:

- *Soundex blocking*: pairs whose last names encode to the same four-character phonetic code. "Garcia" and "Garsia" both encode to "G620", so they remain candidates even after the typo.
- *Birth-year blocking*: pairs who share the same four-digit birth year.

A *union* of the two keys is used: a pair qualifies for comparison if it passes *either* key. This increases recall (fewer missed true matches) at the cost of somewhat more candidate pairs.

The Soundex function (in `examples/chapter-05/04_blocking.py`) maps a last name to a four-character phonetic code: first letter plus three digits encoding consonant groups. "Garcia" and "Garsia" both encode to "G620". The birth-year function simply extracts the first four characters of the DOB string.

Candidate pairs are the union of all pairs that share a Soundex key or a birth year. See `04_blocking.py` for the full implementation.

Typical results on the 200-person synthetic dataset:

| Metric | Value |
|--------|-------|
| Total possible pairs (no blocking) | 40,000 |
| Candidate pairs after blocking | ~1,400 |
| Reduction | >96% |
| True matches surviving blocking | ~178 / 180 |
| Blocking recall | ~99% |

The blocking recall matters. If 2% of true matches are eliminated before scoring begins, no amount of classifier tuning can recover them.

### The blocking trade-off

Blocking is a recall-vs-cost trade-off. Two failure modes pull in opposite directions.

*Too aggressive blocking*: the comparison space is small and computation is fast, but true matches are lost when their blocking keys disagree due to noise. A person whose last name was abbreviated ("M. Garcia" vs "Garcia") may never appear in the same Soundex block as their true match. These are permanent false negatives — no scoring step can recover a pair that was blocked away.

*Too conservative blocking*: more true matches survive, but the candidate set grows. Comparison cost grows quadratically with dataset size: doubling the candidate set doubles the number of feature vectors to compute and the time to score them.

The choice depends on the cost of false negatives. For coverage estimates — counting how many people lack health coverage, or how many businesses filed taxes — missing a match means undercounting. Undercounting is usually the error that program managers care about most. In these contexts, err toward conservative blocking and accept a larger candidate set.

A well-designed blocking strategy can reduce the comparison space by over 99% while maintaining high blocking recall, though the achievable trade-off depends on data quality and the blocking keys chosen (Steorts, Ventura, Sadinle & Fienberg, 2014). The blocking recall can be estimated by comparing the candidate set against a gold standard sample of known true pairs.

---

## 6. Record linkage errors and their downstream effects

Linkage errors are not random. They have systematic effects on analysis that depend on which type of error occurs and which subgroups are affected.

### False matches

A false match occurs when two records for different people are accepted as a link. Their demographic attributes get merged. If person A earns $30,000 and person B earns $90,000, a false match that combines them may produce an estimate of $60,000 for the merged record — or one person's attributes may silently override the other's.

In a household linkage graph, a false match can pull two previously unconnected households into the same component. Downstream analyses that aggregate at the household level — income, family composition, program participation — become unreliable for these clusters.

False matches are most likely when multiple people share similar names and dates of birth. Common names (Smith, Garcia, Lee) in dense urban areas with tight age distributions are higher-risk. The cluster size distribution is the first diagnostic: clusters of size 3+ are the primary flag for false matches in a two-source linkage.

### Missed matches

A missed match occurs when two records for the same person are rejected as a non-link. This person appears twice in the linked dataset: once in each source, without a connection. Coverage counts are wrong. If you are estimating how many people participated in both programs, missed matches make the overlap count too low.

Missed matches are not random. They are systematically more likely for records with harder-to-match fields: names with non-standard transliterations, addresses with non-standard formatting, dates of birth that were entered differently across systems. Subgroups who are more likely to appear in administrative data with data entry variation — recent immigrants, people with non-anglicized names, people who have moved frequently — are systematically undercounted in linked analyses.

When a linked file shows lower match rates for certain demographic or geographic subgroups, this is a data quality finding, not a statistical artifact.

### Transitive closure errors

The graph structure creates a failure mode called *transitive closure error*. Suppose:
- A matches B with probability 0.80 (accepted)
- B matches C with probability 0.80 (accepted)
- But A and C are different people

Accepting both links creates a cluster of three: A, B, and C are now treated as one entity. Even if each individual link seems acceptable at the 0.80 threshold, the chain as a whole may be a false merge.

One common approach is to find the *maximum-weight spanning tree* of each cluster: keep only the highest-probability edges necessary to connect the cluster, which prunes the weakest false-match chain links while preserving the strongest true-match links. `examples/chapter-05/06_linkage_graph.py` demonstrates this with the `nx.maximum_spanning_tree` function.

---

## 7. Evaluating a linked dataset

When you receive a linked file from another program or contractor, the cluster size distribution is not enough. Before using the data, work through this checklist.

**Checklist: evaluating a linked dataset**

1. What was the match rate? (What fraction of Source A records found a match in Source B?)
2. How was the false positive rate estimated? (Manual review sample? Gold standard comparison?)
3. How were ambiguous links resolved? (One-to-many matches, cyclical matches, transitive closure)
4. What blocking strategy was used? (What pairs were never compared — and is that defensible given the false negative cost?)
5. Was the linkage validated against a ground truth dataset?
6. What is the cluster size distribution? (Clusters of size 3+ are a red flag in most applications)
7. Are there subgroup differences in match rates? (Demographic or geographic bias in linkage quality)

Questions 4 and 7 are the ones most often skipped in documentation. If the data supplier cannot answer what pairs were never compared, you cannot assess whether the missed-match rate is acceptable for your use case.

---

## 8. Comparison features and scoring

`examples/chapter-05/05_comparison_scoring.py` computes feature vectors for each candidate pair and trains a logistic regression classifier.

The feature distribution plot — showing histograms of name_sim, dob_sim, and addr_sim separately for true matches and non-matches — is diagnostic. Features where the two distributions overlap heavily contribute little discriminating power. In this synthetic dataset, name similarity and DOB agreement are the strongest signals; address similarity has more overlap because addresses change more frequently than names.

The logistic regression classifier is trained on 70% of candidate pairs and evaluated on 30%. Class weights are balanced to handle the heavy imbalance: in the candidate set, non-matches outnumber true matches by roughly 7 to 1.

The full training and evaluation code is in `examples/chapter-05/05_comparison_scoring.py`. Class weights are balanced (`class_weight="balanced"`) to handle the roughly 7-to-1 non-match majority in the candidate set.

The reported precision and recall refer to the *candidate set*, not the original dataset. Recall at the classifier stage is only meaningful relative to blocking recall: if blocking eliminated 5% of true matches, the maximum possible end-to-end recall is 95%, regardless of how good the classifier is.

---

## 9. Graph structure in survey estimation

### 9.1 Household clustering and design effects

Survey methodologists have long known that the response decision for one household member affects others. If you model persons as independent rows, you underestimate standard errors.

Within a household, the latent response propensity is shared (Groves & Couper, 1998): all members face the same interviewer, the same day's mood, and the same household-level circumstances (language barriers, distrust of government data collection, health events). This creates *intraclass correlation* (ICC): responses within a cluster are more similar than responses drawn at random from the population.

The *design effect* (DEFF; Kish, 1965) quantifies the variance inflation:

$$\text{DEFF} = 1 + (\bar{n} - 1) \times \text{ICC}$$

where $\bar{n}$ is the mean cluster size and ICC is the intraclass correlation. A DEFF of 1.4 means that the effective sample size is only $N / 1.4 \approx 71\%$ of the nominal sample. Standard errors that ignore clustering are too small by a factor of $\sqrt{1.4} \approx 1.18$.

`examples/chapter-05/07_survey_effects.py` simulates this effect with 300 households drawn from a Beta(4,2) propensity distribution and computes the resulting ICC and DEFF.

When using graph-linked data in predictive models, `GroupKFold` (introduced in Chapter 2) must be applied at the household or primary sampling unit (PSU) level, not the individual level. Splitting at the person level leaks within-household correlation into the validation set and produces optimistically biased estimates of out-of-sample accuracy.

The fix is `GroupKFold(n_splits=5)` with `groups=household_ids` — not person IDs. Chapter 2 covers cross-validation strategy in detail; the key addition here is that the grouping variable must be the cluster identifier from the linkage graph, not the row identifier.

### 9.2 Geographic spillover

Economic shocks, policy changes, and environmental events spread across geographic boundaries. A plant closure in one county affects unemployment in adjacent counties through commuting patterns. A survey interviewer assigned to multiple adjacent tracts introduces correlated measurement error across those tracts.

Treating counties or tracts as independent units in an analysis ignores this spatial dependence and understates the variance of area-level estimates. The geographic grid simulation in `examples/chapter-05/07_survey_effects.py` illustrates how a shock decays with Manhattan distance from its epicenter: the center county shows full shock intensity while corner counties show near-zero values, with a smooth gradient across neighbors.

Standard survey variance estimation methods — jackknife, bootstrap replicate weights, balanced repeated replication — account for geographic clustering through the PSU structure of the sample design. When using administrative data or linked files that lack these design weights, geographic autocorrelation must be addressed explicitly.

---

## 10. In-class activity

You receive two extracts from different administrative systems for the same county: a public assistance roll (`admin_a`) and a healthcare enrollment file (`admin_b`). There are 150 people who appear in both files, plus 30 additional records in `admin_b` with no corresponding record in `admin_a`.

`examples/chapter-05/08_exercise.py` generates these datasets and contains the full solution.

**Discussion questions** (no code required):

1. You receive a linked file with this cluster size distribution:

   | Cluster size | Count |
   |-------------|-------|
   | 1 (singleton) | 180 records |
   | 2 (matched pair) | 52 clusters |
   | 3 | 8 clusters |
   | 4+ | 3 clusters |

   Which clusters would you flag for manual review? Why? What would you check in each flagged cluster?

2. The match rate for Hispanic surnames is 14% lower than for non-Hispanic surnames. What might explain this? What would you recommend to the program manager before using this linked file for analysis?

3. Your blocking strategy reduces 40,000 possible pairs to 1,200 candidates. The match rate within the candidate set is 82%. Estimate how many true matches were likely missed by the blocking step. What additional information would you need to make this estimate precise?

4. *Optional — run the full pipeline*: Using `examples/chapter-05/08_exercise.py`, run the complete pipeline and compare your blocking recall, precision, and recall values to the reference solution.

---

## Key takeaways for survey methodology

- *Many datasets in federal statistics have graph structure.* Households, employer-employee relationships, geographic hierarchies, and administrative record links are all graphs. Ignoring this structure means ignoring information.
- *Record linkage is the foundation of data integration.* Understanding blocking, comparison, scoring, and clustering is essential for evaluating any linked data product your program receives or produces.
- *False matches and missed matches have different costs.* A false match merges two entities and introduces correlated errors. A missed match loses coverage. The threshold controls this trade-off, but blocking errors are unrecoverable.
- *Missed matches are not random.* Subgroups with harder-to-match fields — non-anglicized names, non-standard addresses, transposed dates — are systematically undercounted in linked analyses. Subgroup match-rate comparison is a required quality check.
- *The cluster size distribution is the primary diagnostic.* Clusters of size 2 are expected. Clusters of size 3+ warrant investigation for false match chains.
- *Graph structure affects standard errors.* Household clustering creates intraclass correlation that inflates variance. Ignoring it produces standard errors that are too small. GroupKFold at the household/PSU level is the correct validation approach for models built on clustered survey data.

```{admonition} How to explain these methods to leadership
:class: tip
**What is record linkage?** We have two data files that describe the same people, but they don't have a shared unique ID. Record linkage finds the pairs of records (one from each file) that refer to the same person, using information like name, date of birth, and address.

**Why can't we just match on names?** Names vary: "Maria Garcia" might appear as "M. Garcia" or "Maria Garica" (typo). Date of birth gets transposed. Address abbreviations differ. We need to measure similarity rather than require exact equality, then set a threshold for what counts as a match.

**How do we know how well it worked?** If we have some pairs where we know the true answer (gold standard), we can measure precision (what fraction of accepted matches are correct) and recall (what fraction of true matches we found). The threshold controls the trade-off between these two. An equally important question is whether match quality is uniform across demographic subgroups — lower match rates in some groups mean those groups are undercounted in linked analyses.

**What does the graph add?** After linking, records cluster into groups. Each group should represent one real person. Groups with more than two records may indicate false positives (two different people mistakenly linked) and require manual review. Visualizing the linkage graph makes these ambiguous cases immediately visible.
```
