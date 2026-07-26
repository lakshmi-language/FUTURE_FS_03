# Resume Screening System

A lightweight, dependency-tolerant pipeline for ranking candidate resumes against a job description using TF-IDF text similarity combined with skill-taxonomy matching.

## What it does

1. **Cleans text** — lowercases, strips URLs, emails, phone numbers, and punctuation.
2. **Extracts skills** — matches resume/JD text against a configurable skill taxonomy (word-boundary aware, handles multi-word skills like "machine learning").
3. **Ranks candidates** — computes a blended score per resume:
   `final_score = 0.6 * TF-IDF cosine similarity + 0.4 * skill match score`
4. **Reports skill gaps** — lists JD-required skills missing from each resume.

Tokenization prefers spaCy (with lemmatization) when available, falls back to NLTK stopwords, and falls back further to a built-in stopword list if neither is installed — so the script runs standalone even without full NLP dependencies.

## Requirements

```
pandas
scikit-learn
spacy
nltk
```

Install with:
```bash
pip install -r requirements.txt
```

Optional (for full spaCy lemmatization support):
```bash
python -m spacy download en_core_web_sm
```

## Usage

Run the built-in demo (ranks 3 sample resumes against a sample Data Scientist JD):
```bash
python resume_screening.py
```

Or use it in your own code:
```python
from resume_screening import ResumeScreeningSystem

system = ResumeScreeningSystem()  # or pass a custom skill_taxonomy list

resumes = {
    "Candidate A": "resume text here...",
    "Candidate B": "resume text here...",
}
job_description = "job description text here..."

results = system.rank_candidates(resumes, job_description)
print(results)
```

`rank_candidates` returns a pandas DataFrame ranked by `final_score`, with columns for `text_similarity`, `skill_match_score`, `matched_skills`, and `missing_skills`.

## Customizing the skill taxonomy

Pass your own list of skills when constructing the system to tailor it to a specific role or industry:
```python
system = ResumeScreeningSystem(skill_taxonomy=["python", "aws", "terraform", "ci/cd"])
```

If omitted, a default general tech-skill taxonomy (languages, ML/data tools, cloud, soft skills) is used.

## Files

| File | Purpose |
|---|---|
| `resume_screening.py` | Core `ResumeScreeningSystem` class + standalone demo |
| `requirements.txt` | Python dependencies |
