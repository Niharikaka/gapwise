# AI-demand classifier: pilot model card

## Intended use

Screen Singapore job descriptions for advertised AI skill demand and compare plausible course opportunities. Explicit preferred qualifications count as hiring signals. Potential adoption applications remain separate from observed demand. The model is not evidence that an employer currently uses AI, that workers lack skills, or that a course will sell.

## Components

- **Legacy comparator:** original shared-dictionary job flag.
- **Evidence screen:** contextual explicit/implied/ambiguous/no-evidence baseline, augmented by accepted DeepSeek reviews where available.
- **AI reviewer:** DeepSeek Flash, JSON extraction with task quotations, requirement strength, maturity and separate adoption hypotheses. Invalid evidence is rejected, with a bounded repair attempt.
- **Local student:** word/bigram TF-IDF and class-balanced logistic regression. It predicts binary expanded demand. It does not predict a verified explicit/implied distinction or a verified maturity level.

## Data and separation

The pilot accepted 972 unique job input texts and 79 unique course input texts under the current prompt. Exact-equivalent text reviews cover additional duplicate postings/catalogue entries; these are identified separately from unique paid requests. All 94,316 jobs receive local model scores. This is not a full-population LLM review.

Student training uses 432 distinct description groups, with 145 separate development examples. The existing 400 human-labelled jobs and their identical descriptions are excluded from training and development. The decision threshold is selected on the teacher-labelled development split. No human benchmark labels are used as model inputs or training labels. The old benchmark had previously been inspected, so it is a reused benchmark, not a new blinded test.

## Results against the existing weighted human benchmark

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| Original dictionary | 0.543 | 0.245 | 0.338 |
| Local AI-supervised student | 0.633 | 0.476 | 0.544 |

The paired stratified bootstrap interval for the F1 difference is approximately **-0.022 to +0.385**. The point estimate improves, but this sample does not establish a statistically clear population improvement. These intervals are conditional on the existing labels, not uncertainty over label correctness.

On the teacher-labelled development subset, student F1 agreement is about 0.810 versus 0.757 for the original dictionary. This measures agreement with the teacher. It is also the development split used for threshold selection, so it is not an independent final performance estimate.

## Important limitations

- Existing human positives include generic software, desktop support and infrastructure roles without explicit AI requirements. Their broader inferred-adoption construct differs from strict observed demand. Both labels and all disagreements are preserved.
- A single AI reviewer can make systematic mistakes. Quotation validation establishes traceability, not semantic truth.
- The student can predict a positive label without identifying defensible task evidence. Its output must stay separate from explicit evidence-based classification.
- Scores are uncalibrated. The near-threshold band is a heuristic, not a confidence interval.
- The training sample was balanced on the old screen and is not a random population sample. Deployment prevalence and probability calibration require further evaluation.
- Duplicate description groups are kept apart, but broader recruiter/employer template similarity can still create dependence.
- Course descriptions differ from actual teaching. University availability is not CET capacity; only NUS has a comparable current-offering filter.

## Operational controls

The API budget is durable and cumulative across prompt revisions, retries and reruns. The authorised pilot cap is $2, with conservative pricing reservations. The temporary credential was removed; subsequent API runs require a locally configured environment variable. No key is stored in project source.

The local model is not automatically promoted over the original comparator. All four analytical scenarios remain available. Use `outputs/model_comparison.json`, `outputs/benchmark_uncertainty.json`, and the scenario manifests as the authoritative numerical record.
