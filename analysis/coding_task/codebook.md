# Codebook: taxonomy coding of AI-plagiarism-detection papers

Revision R2. These definitions replace the R1 versions, which were not mutually
exclusive and produced Cohen's kappa = 0.44 on the methodology dimension.

You are coding individual papers, one at a time, from the title, author/index
keywords, document type and abstract. Assign **exactly one** category on each of
the two dimensions. Never leave a paper uncoded: if the evidence is thin, apply
the tie-break rule and record lower confidence.

---

## Dimension 1 — Research orientation

What the paper is *about*: the problem it addresses.

**technical** — Detection methods and systems as the object of study.
Developing, improving, benchmarking or comparing algorithms, classifiers,
language models, embeddings, stylometric or watermarking techniques,
source-code or image similarity methods. The paper's contribution is knowledge
about how detection *works*.

**pedagogical** — Teaching, learning and assessment as the object of study.
Student behaviour, beliefs or performance; assessment and curriculum design;
classroom practice; instructor responses; academic-writing instruction. The
paper's contribution is knowledge about how people *teach and learn* in the
presence of AI.

**governance** — Policy, ethics, law and institutions as the object of study.
Institutional integrity policy, regulation, copyright and authorship, research
ethics, scholarly publishing and peer review, editorial standards, misconduct
procedures. The paper's contribution is knowledge about how the activity should
be *governed*.

**Tie-break.** Many papers touch two orientations. Ask: *if this paper's finding
turned out to be wrong, whose work would be most affected — engineers,
educators, or policymakers?* Code for that audience. Only if genuinely balanced,
apply the precedence technical > pedagogical > governance.

Note: nearly every paper in this corpus mentions detection, AI and plagiarism,
because the corpus was retrieved by searching for those terms. **Their presence
carries no information.** Do not code a paper "technical" merely because it
mentions detection or AI.

---

## Dimension 2 — Methodological approach

What the paper *did*: its primary evidence type. Code the dominant contribution,
not everything the paper contains.

**computational** — The primary evidence is produced by building or running a
computational artefact. The paper proposes, implements, trains, fine-tunes or
benchmarks a model, classifier, algorithm, architecture, pipeline, dataset or
tool, and reports quantitative performance (accuracy, precision/recall, F1,
AUC, error rates) on documents or text.

**empirical** — The primary evidence is data the authors collected about people
or documents, analysed without a new computational artefact. Surveys,
questionnaires, interviews, focus groups, classroom or laboratory experiments
with participants, case studies, document or policy content analysis. Reports
what people did, said, believed, or what a body of documents contains.

**conceptual** — No new data and no new artefact. The paper argues, reviews,
synthesises or theorises: literature reviews, position and perspective pieces,
commentaries, editorials, legal or ethical analyses, proposed frameworks and
recommendations derived by reasoning rather than measurement.

### Decision procedure (apply in order)

1. **Did the authors build or run a computational system and report its
   performance?** If yes → `computational`. (A paper that *evaluates existing
   detectors* by running them on texts and reporting accuracy is
   `computational`, even if the framing is educational.)
2. **Otherwise, did the authors collect data from people or a document
   corpus and analyse it?** If yes → `empirical`. (A survey of student attitudes
   towards detectors is `empirical`. A study that asks humans to judge AI-written
   text and reports their accuracy is `empirical`.)
3. **Otherwise** → `conceptual`.

### Boundary cases — decide these consistently

- **Evaluating commercial detectors** (running Turnitin/GPTZero on sample texts
  and reporting detection rates) → `computational`. Evidence comes from running
  a system, even though the authors did not build it.
- **Survey plus a small classifier demo** → whichever the abstract presents as
  the study's main result. If the abstract leads with the survey, → `empirical`.
- **Literature review that counts papers** (bibliometric or systematic review)
  → `conceptual`, unless it trains or applies a model to the corpus, in which
  case → `computational`.
- **Proposes a framework or set of guidelines with no data** → `conceptual`.
- **Case study of an institution's policy response** → `empirical` if it reports
  documents or interviews collected by the authors; `conceptual` if it is
  argument illustrated by an example.
- **Editorials, letters, notes** → almost always `conceptual`.

---

## Output

For every one of the 100 papers, return:

- `paper_id` — exactly as given
- `orientation` — one of `technical`, `pedagogical`, `governance`
- `methodology` — one of `computational`, `empirical`, `conceptual`
- `confidence` — `high`, `medium` or `low`
- `note` — at most 12 words, only when confidence is `low`

Code every paper independently, on its own merits. Do not try to balance the
category counts, and do not assume papers near each other in the list are
related — the order carries no meaning.
