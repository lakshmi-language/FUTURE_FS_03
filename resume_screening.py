import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Optional NLP backends. Falls back gracefully if spaCy model isn't installed,
# so the script still runs standalone.
# ---------------------------------------------------------------------------
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        # Model not downloaded: `python -m spacy download en_core_web_sm`
        nlp = None
        SPACY_AVAILABLE = False
except ImportError:
    nlp = None
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    try:
        STOPWORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOPWORDS = set(stopwords.words("english"))
except (ImportError, LookupError, OSError):
    # nltk not installed or no network to fetch corpora -> use a built-in
    # fallback stopword list so the script still runs standalone.
    STOPWORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
        "your", "yours", "he", "him", "his", "she", "her", "it", "its",
        "they", "them", "their", "what", "which", "who", "this", "that",
        "these", "those", "am", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "having", "do", "does", "did",
        "doing", "a", "an", "the", "and", "but", "if", "or", "because",
        "as", "of", "at", "by", "for", "with", "about", "against",
        "between", "into", "through", "during", "before", "after",
        "above", "below", "to", "from", "up", "down", "in", "out", "on",
        "off", "over", "under", "again", "further", "then", "once",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "s", "t", "can", "will", "just", "don", "should", "now",
    }


class ResumeScreeningSystem:
    """
    End-to-end resume screening pipeline:
        clean -> extract skills -> vectorize -> rank -> gap analysis
    """

    def __init__(self, skill_taxonomy=None):
        """
        skill_taxonomy: list of known skills used for extraction/matching.
        If None, a default general tech-skill list is used.
        """
        self.skill_taxonomy = skill_taxonomy or self._default_skill_taxonomy()
        # Normalize taxonomy for matching (lowercase)
        self._skill_lookup = {s.lower(): s for s in self.skill_taxonomy}
        self.vectorizer = TfidfVectorizer(stop_words="english")

    # ------------------------------------------------------------------
    # 1. Text cleaning & parsing
    # ------------------------------------------------------------------
    @staticmethod
    def clean_text(text: str) -> str:
        """Lowercase, strip punctuation/extra whitespace, remove noise."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
        text = re.sub(r"\S+@\S+", " ", text)                 # emails
        text = re.sub(r"\+?\d[\d\-\s]{7,}\d", " ", text)      # phone numbers
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize(self, text: str):
        """Tokenize using spaCy if available, else simple NLTK-style split."""
        cleaned = self.clean_text(text)
        if SPACY_AVAILABLE:
            doc = nlp(cleaned)
            tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct]
        else:
            tokens = [w for w in cleaned.split() if w not in STOPWORDS]
        return tokens

    # ------------------------------------------------------------------
    # 2. Skill extraction & matching
    # ------------------------------------------------------------------
    def extract_skills(self, text: str) -> list:
        """
        Extract known skills mentioned in a text (resume or job description)
        using keyword/phrase matching against the skill taxonomy.
        Handles multi-word skills (e.g. 'machine learning').
        """
        cleaned = self.clean_text(text)
        found = set()
        for skill_lower, skill_original in self._skill_lookup.items():
            # word-boundary match so 'r' doesn't match inside 'research'
            pattern = r"\b" + re.escape(skill_lower) + r"\b"
            if re.search(pattern, cleaned):
                found.add(skill_original)
        return sorted(found)

    def skill_match_score(self, resume_skills: list, jd_skills: list) -> float:
        """Fraction of job-description skills present in the resume (0-1)."""
        if not jd_skills:
            return 0.0
        matched = set(s.lower() for s in resume_skills) & set(s.lower() for s in jd_skills)
        return round(len(matched) / len(jd_skills), 3)

    def skill_gaps(self, resume_skills: list, jd_skills: list) -> list:
        """Skills required by the JD but missing from the resume."""
        resume_lower = set(s.lower() for s in resume_skills)
        return sorted([s for s in jd_skills if s.lower() not in resume_lower])

    # ------------------------------------------------------------------
    # 3. Candidate ranking (TF-IDF + cosine similarity to job description)
    # ------------------------------------------------------------------
    def rank_candidates(self, resumes: dict, job_description: str) -> pd.DataFrame:
        """
        resumes: dict {candidate_name: resume_text}
        job_description: the target job description text

        Returns a DataFrame ranked by a blended score:
            final_score = 0.6 * text_similarity + 0.4 * skill_match_score
        """
        jd_skills = self.extract_skills(job_description)
        names = list(resumes.keys())
        texts = [self.clean_text(resumes[n]) for n in names]
        jd_clean = self.clean_text(job_description)

        # TF-IDF similarity between each resume and the JD
        corpus = texts + [jd_clean]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        jd_vector = tfidf_matrix[-1]
        resume_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(resume_vectors, jd_vector).flatten()

        rows = []
        for i, name in enumerate(names):
            resume_skills = self.extract_skills(resumes[name])
            match_score = self.skill_match_score(resume_skills, jd_skills)
            gaps = self.skill_gaps(resume_skills, jd_skills)
            text_sim = round(float(similarities[i]), 3)
            final_score = round(0.6 * text_sim + 0.4 * match_score, 3)

            rows.append({
                "candidate": name,
                "text_similarity": text_sim,
                "skill_match_score": match_score,
                "final_score": final_score,
                "matched_skills": ", ".join(sorted(
                    set(s.lower() for s in resume_skills) & set(s.lower() for s in jd_skills)
                )),
                "missing_skills": ", ".join(gaps),
            })

        df = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
        df.index += 1  # rank starts at 1
        df.index.name = "rank"
        return df

    # ------------------------------------------------------------------
    # Default skill taxonomy (can be replaced/extended per role)
    # ------------------------------------------------------------------
    @staticmethod
    def _default_skill_taxonomy():
        return [
            "python", "java", "c++", "sql", "r", "javascript",
            "machine learning", "deep learning", "nlp", "natural language processing",
            "data analysis", "data visualization", "pandas", "numpy", "scikit-learn",
            "tensorflow", "pytorch", "spacy", "nltk", "power bi", "tableau",
            "excel", "aws", "azure", "gcp", "docker", "kubernetes",
            "git", "rest api", "flask", "django", "communication", "leadership",
            "project management", "agile", "scrum", "statistics", "etl",
        ]


# ---------------------------------------------------------------------------
# Standalone demo run with sample data
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    job_description = """
    We are hiring a Data Scientist with strong experience in Python,
    machine learning, and NLP. The candidate should be comfortable with
    pandas, scikit-learn, and SQL for data analysis. Experience with
    TensorFlow or PyTorch and knowledge of Agile/Scrum practices is a plus.
    Good communication skills required.
    """

    sample_resumes = {
        "Anita Rao": """
            Data scientist with 3 years experience in Python, pandas, and
            scikit-learn. Built NLP models using spaCy and NLTK for text
            classification. Familiar with SQL and Tableau. Strong communicator,
            worked in Agile teams.
        """,
        "Rahul Mehta": """
            Software engineer skilled in Java and C++. Some exposure to
            machine learning coursework using Python. Familiar with Git and
            REST APIs. No direct NLP or SQL project experience.
        """,
        "Priya Nair": """
            Machine learning engineer with deep learning expertise in
            TensorFlow and PyTorch. Strong Python and SQL background.
            Experience with NLP projects using spaCy. Led a small team using
            Scrum methodology. Excellent communication skills.
        """,
    }

    system = ResumeScreeningSystem()

    print("=== Job Description Skills Detected ===")
    print(system.extract_skills(job_description))
    print()

    print("=== Candidate Ranking ===")
    results = system.rank_candidates(sample_resumes, job_description)
    print(results.to_string())

    print("\n=== Skill Gap Report ===")
    for name, resume in sample_resumes.items():
        resume_skills = system.extract_skills(resume)
        jd_skills = system.extract_skills(job_description)
        gaps = system.skill_gaps(resume_skills, jd_skills)
        print(f"{name}: missing -> {gaps if gaps else 'None (fully matched)'}")
