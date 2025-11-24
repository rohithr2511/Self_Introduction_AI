import re
from collections import Counter
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer

# Cached model loader
@st.cache_resource
def load_models():
    return SentimentIntensityAnalyzer(), SentenceTransformer('all-MiniLM-L6-v2')

sentiment_analyzer, semantic_model = load_models()

# Compact rubric & weights
WEIGHTS = {
    "salutation": 5,
    "keywords": 20,
    "flow": 5,
    "rate": 10,
    "grammar": 15,
    "vocab": 15,
    "filler": 10,
    "sentiment": 10
}

MUST_KEYWORDS = {
    "name": ["name", "myself", "i am", "i'm", "call me"],
    "age": ["age", "years old", "year old"],
    "school": ["school", "studying", "student"],
    "class": ["class", "grade", "standard", "section"],
    "family": ["family", "father", "mother", "parents", "brother", "sister"],
    "hobbies": ["hobby", "hobbies", "enjoy", "like", "love", "interest", "favorite", "favourite", "play"]
}
GOOD_KEYWORDS = {
    "family_details": ["kind", "caring", "loving", "supportive"],
    "location": ["from", "live in", "belong"],
    "goals": ["ambition", "goal", "dream", "want to", "aspire", "hope"],
    "unique": ["fun fact", "special", "unique", "interesting"],
    "achievements": ["achievement", "award", "won", "proud"]
}
FILLERS = ["um", "uh", "like", "you know", "so", "actually", "basically", "right", "i mean", "well", "kinda", "sort of", "okay", "hmm", "ah"]


# Helpers (compact)
def _clean(text):
    return re.sub(r'\s+', ' ', text.strip())

def contains_any(text, keywords):
    t = text.lower()
    return any(k in t for k in keywords)

def count_category_matches(text, categories):
    t = text.lower()
    found = []
    for cat, keys in categories.items():
        if any(k in t for k in keys):
            found.append(cat)
    return found

def ttr_score(text):
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return 0, "No words"
    unique = len(set(words)); total = len(words)
    ttr = unique / total
    if ttr >= 0.75: return 15, f"Excellent - TTR {ttr:.2f} ({unique}/{total})"
    if ttr >= 0.65: return 13, f"Very Good - TTR {ttr:.2f} ({unique}/{total})"
    if ttr >= 0.55: return 11, f"Good - TTR {ttr:.2f} ({unique}/{total})"
    if ttr >= 0.45: return 9, f"Fair - TTR {ttr:.2f} ({unique}/{total})"
    return 7, f"Basic - TTR {ttr:.2f} ({unique}/{total})"

def grammar_score(text):
    # simple major-issue checks -> small penalty per match
    patterns = [ (r'\s+i\s+(?=[a-z])', 0.5), (r'[.!?]\s+[a-z]', 0.5) ]
    errors = 0
    for pat, w in patterns:
        errors += len(re.findall(pat, text)) * w
    # count very short fragments
    fragments = re.split(r'[.!?]+', text)
    for f in fragments:
        if len(f.strip().split()) == 1:
            errors += 0.5
    total_words = max(1, len(text.split()))
    errors_per_100 = (errors / total_words) * 100
    # map to 15
    if errors_per_100 < 1: return 15, "Very few grammar issues"
    if errors_per_100 < 2: return 13, "Minor grammar issues"
    if errors_per_100 < 3: return 11, "Some grammar issues"
    if errors_per_100 < 5: return 9, "Multiple grammar issues"
    return 7, "Needs grammar improvement"

def filler_score(text):
    words = re.findall(r'\b\w+\b', text.lower())
    total = len(words)
    if total == 0:
        return 8, "No words"
    t = ' ' + text.lower() + ' '
    count = sum(t.count(' ' + f + ' ') for f in FILLERS)
    rate = (count / total) * 100
    if rate < 1: return 10, f"Excellent - filler {rate:.1f}% ({count})"
    if rate < 2: return 9, f"Very Good - filler {rate:.1f}% ({count})"
    if rate < 3: return 8, f"Good - filler {rate:.1f}% ({count})"
    if rate < 5: return 6, f"Fair - filler {rate:.1f}% ({count})"
    return 4, f"Needs improvement - filler {rate:.1f}% ({count})"

def salutation_score(text):
    first = '. '.join(text.split('.')[:2]).lower()
    excellent = ["i am excited", "feeling great", "pleased to introduce"]
    good = ["good morning", "good afternoon", "good evening", "good day", "hello everyone", "greetings"]
    normal = ["hi", "hello", "hey"]
    if any(p in first for p in excellent): return 5, "Excellent salutation - enthusiastic"
    if any(p in first for p in good): return 5, "Proper greeting"
    if any(p in first for p in normal): return 4, "Basic greeting"
    return 2, "Started directly"

def sentiment_score(text):
    s = sentiment_analyzer.polarity_scores(text)
    c, pos = s['compound'], s['pos']
    if c >= 0.3 or pos >= 0.2: return 10, f"Positive & engaging (compound {c:.2f})"
    if c >= 0.1 or pos >= 0.1: return 9, f"Friendly (compound {c:.2f})"
    if c >= -0.1: return 8, f"Neutral-positive (compound {c:.2f})"
    if c >= -0.3: return 6, f"Neutral (compound {c:.2f})"
    return 4, f"Could be more positive (compound {c:.2f})"

def speech_rate_score(text, duration_sec):
    if duration_sec <= 0:
        return 8, "Duration not provided"
    words = len(text.split())
    wpm = (words / duration_sec) * 60
    if 100 <= wpm <= 150: return 10, f"Excellent speech rate: {wpm:.1f} WPM"
    if 80 <= wpm < 100 or 150 < wpm <= 170: return 8, f"Good speech rate: {wpm:.1f} WPM"
    if 60 <= wpm < 80 or 170 < wpm <= 190: return 6, f"Acceptable speech rate: {wpm:.1f} WPM"
    return 4, f"Speech rate: {wpm:.1f} WPM (consider adjusting pace)"


# Main analyzer (compact orchestration)
def analyze_transcript(transcript, duration_sec=52):
    text = _clean(transcript)
    results = {
        "overall_score": 0,
        "word_count": len(text.split()),
        "sentence_count": len([s for s in text.split('.') if s.strip()]),
        "criteria_scores": []
    }

    # salutation
    s_score, s_fb = salutation_score(text)
    results["criteria_scores"].append({"criterion": "Salutation", "score": s_score, "max_score": WEIGHTS["salutation"], "weight": WEIGHTS["salutation"], "feedback": s_fb})
    results["overall_score"] += (s_score / 5) * WEIGHTS["salutation"]

    # keywords 
    found_must = count_category_matches(text, MUST_KEYWORDS)
    must_pts = min(len(found_must) * (WEIGHTS["keywords"] / len(MUST_KEYWORDS)), WEIGHTS["keywords"])
    found_good = count_category_matches(text, GOOD_KEYWORDS)
    bonus = min(len(found_good), 5)
    kw_fb = f"Found must-have: {', '.join(found_must) or 'none'}; good extras: {', '.join(found_good) or 'none'}"
    kw_score_total = min(must_pts + bonus, WEIGHTS["keywords"])
    results["criteria_scores"].append({"criterion": "Content & Keywords", "score": kw_score_total, "max_score": WEIGHTS["keywords"], "weight": WEIGHTS["keywords"], "feedback": kw_fb})
    results["overall_score"] += kw_score_total

    # flow
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    name_early = any(('name' in s.lower() or 'myself' in s.lower() or 'i am' in s.lower()) for s in sentences[:2])
    has_closing = any(w in sentences[-1].lower() for w in ['thank', 'thanks', 'listening', 'attention']) if sentences else False
    has_middle = len(sentences) >= 4
    flow_pts = (2 if name_early else 0) + (2 if has_middle else 0) + (1 if has_closing else 0)
    results["criteria_scores"].append({"criterion": "Flow & Structure", "score": flow_pts, "max_score": WEIGHTS["flow"], "weight": WEIGHTS["flow"], "feedback": f"{'good opening, ' if name_early else ''}{'middle content, ' if has_middle else ''}{'closing' if has_closing else ''}".strip(' ,') or "Basic flow"})
    results["overall_score"] += flow_pts

    # speech rate
    rate_pts, rate_fb = speech_rate_score(text, duration_sec)
    # rate_pts is out of 10; weight is 10
    results["criteria_scores"].append({"criterion": "Speech Rate", "score": rate_pts, "max_score": WEIGHTS["rate"], "weight": WEIGHTS["rate"], "feedback": rate_fb})
    results["overall_score"] += rate_pts

    # grammar
    gram_pts, gram_fb = grammar_score(text)
    results["criteria_scores"].append({"criterion": "Grammar & Language", "score": gram_pts, "max_score": WEIGHTS["grammar"], "weight": WEIGHTS["grammar"], "feedback": gram_fb})
    results["overall_score"] += gram_pts

    # vocab (TTR mapped to 15)
    vocab_pts, vocab_fb = ttr_score(text)
    results["criteria_scores"].append({"criterion": "Vocabulary Richness", "score": vocab_pts, "max_score": WEIGHTS["vocab"], "weight": WEIGHTS["vocab"], "feedback": vocab_fb})
    results["overall_score"] += vocab_pts

    # filler
    fill_pts, fill_fb = filler_score(text)
    results["criteria_scores"].append({"criterion": "Clarity (Filler Words)", "score": fill_pts, "max_score": WEIGHTS["filler"], "weight": WEIGHTS["filler"], "feedback": fill_fb})
    results["overall_score"] += fill_pts

    # sentiment
    sent_pts, sent_fb = sentiment_score(text)
    results["criteria_scores"].append({"criterion": "Engagement & Positivity", "score": sent_pts, "max_score": WEIGHTS["sentiment"], "weight": WEIGHTS["sentiment"], "feedback": sent_fb})
    results["overall_score"] += sent_pts

    # Normalize overall to 0-100 (WEIGHTS total is 100 when each criterion sum equals its weight)
    results["overall_score"] = float(min(max(results["overall_score"], 0), 100))
    return results


# Streamlit UI
st.set_page_config(page_title="Transcript Scorer", page_icon="📝", layout="wide")

col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("assests/Gemini_Generated_Image_8p83py8p83py8p83.png", use_container_width=True)
with col2:
    st.title("Self-Introduction Transcript Scorer")

st.markdown("Analyze self-introduction transcripts based on content, language, clarity, and engagement criteria.")

with st.sidebar:
    st.header("ℹ️ Scoring Guide")
    st.markdown("""
    **Score Ranges:**
    - 85-100: Excellent
    - 70-84: Good
    - 55-69: Fair
    - Below 55: Needs Improvement

    **Tips for High Scores:**
    - Include: name, age, class, school, family, hobbies
    - Start with a greeting
    - End with "thank you"
    - Speak at 100-150 WPM
    - Use varied vocabulary
    - Be positive and engaging
    """)

col1, col2 = st.columns([3, 1])
with col1:
    transcript_input = st.text_area(
        "Enter or paste the transcript here:",
        height=250,
        placeholder="Hello everyone, myself Muskan, studying in class 8th B section from Christ Public School..."
    )

with col2:
    duration = st.number_input("Duration (seconds):", min_value=1, value=52, step=1)
    uploaded_file = st.file_uploader("Or upload a text file:", type=['txt'])
    if uploaded_file:
        transcript_input = uploaded_file.read().decode('utf-8')

if st.button("Score Transcript", type="primary", use_container_width=True):
    if transcript_input and transcript_input.strip():
        with st.spinner("Analyzing transcript..."):
            results = analyze_transcript(transcript_input, duration)

        st.markdown("---")
        overall = results['overall_score']
        if overall >= 85:
            score_color = "🟢"; grade = "Excellent!"
        elif overall >= 70:
            score_color = "🟡"; grade = "Good"
        elif overall >= 55:
            score_color = "🟠"; grade = "Fair"
        else:
            score_color = "🔴"; grade = "Needs Improvement"

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Overall Score", f"{overall:.1f}/100")
        with c2: st.metric("Grade", f"{score_color} {grade}")
        with c3: st.metric("Word Count", results['word_count'])
        with c4: st.metric("Sentences", results['sentence_count'])

        st.markdown("---")
        st.subheader("Detailed Criteria Scores")
        for criterion in results['criteria_scores']:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write(f"**{criterion['criterion']}**")
                # show progress relative to the criterion max
                max_s = criterion.get("max_score", criterion.get("weight", 10))
                progress_value = min(max(criterion['score'] / max_s, 0.0), 1.0)
                st.progress(progress_value)
                st.caption(criterion.get("feedback", ""))
            with col_b:
                st.metric("Score", f"{criterion['score']:.1f}/{criterion.get('max_score', criterion.get('weight', 10))}")
                pct = (criterion['score'] / criterion.get('max_score', criterion.get('weight', 10))) * 100 if criterion.get('max_score', None) else 0
                st.caption(f"{pct:.0f}%")
            st.markdown("")

        st.markdown("---")
        with st.expander("View JSON Output"):
            st.json(results)
    else:
        st.error("Please enter a transcript to analyze!")

with st.expander("View Sample Transcript (Expected Score: 80-85)"):
    st.text("""Hello everyone, myself Muskan, studying in class 8th B section from Christ Public School.
I am 13 years old. I live with my family. There are 3 people in my family, me, my mother and my father.
One special thing about my family is that they are very kind hearted to everyone and soft spoken. 
One thing I really enjoy is playing cricket and taking wickets. A fun fact about me is that I see in mirror and talk by myself. 
One thing people don't know about me is that I once stole a toy from one of my cousin.
My favorite subject is science because it is very interesting. Through science I can explore the whole world and make discoveries and improve the lives of others.
Thank you for listening.""")

st.markdown("---")
st.caption("This scorer uses NLP techniques including sentiment analysis, vocabulary analysis, and pattern matching to evaluate transcripts.")
