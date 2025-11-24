# 📝 Self-Introduction Transcript Scorer

An intelligent, NLP-powered Streamlit application designed to evaluate and score self-introduction transcripts. This tool uses sentiment analysis, vocabulary diversity metrics, and pattern matching to provide immediate, detailed feedback to students or individuals practicing their introductions.

## 🚀 Features

* **Automated Scoring:** Instantly calculates a score out of 100 based on a weighted rubric.
* **NLP Analysis:** Utilizes **VADER Sentiment Analysis** to gauge engagement and positivity.
* **Vocabulary Metrics:** Calculates Type-Token Ratio (TTR) to assess vocabulary richness.
* **Speech Rate Calculation:** Estimates Words Per Minute (WPM) based on text length and user-provided duration.
* **Detailed Feedback:** Provides specific feedback on:
    * Salutations & Closings
    * Grammar & Sentence Structure
    * Filler Word Usage (e.g., "um", "uh", "like")
    * Content Completeness (Name, Age, Hobbies, Family)
* **Interactive UI:** Built with Streamlit for a responsive, user-friendly experience.


## Live Demo

🔗 Deployed on Streamlit Cloud:

https://selfintroductionai-ez9xzwbuk2ddntlyufjbzq.streamlit.app/


## 🛠 Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/Self_Introduction_AI.git
cd Self_Introduction_AI

2️⃣ Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit app
streamlit run app.py



## 🛠️ Tech Stack
-Python 3.8+

-Streamlit: For the web interface.

-VADER Sentiment: For analyzing tone and emotion.

-Sentence-Transformers: Loaded for semantic capabilities.

-Regex (re): For pattern matching (grammar, keywords, fillers).

## 📘 How Scoring Works
| Criterion           | Weight | Description                      |
| ------------------- | ------ | -------------------------------- |
| Salutation          | 5      | Detects proper greeting start    |
| Content & Keywords  | 20     | Checks required & bonus keywords |
| Flow & Structure    | 5      | Opening, body, and closing       |
| Speech Rate         | 10     | WPM calculation                  |
| Grammar             | 15     | Error density & common mistakes  |
| Vocabulary Richness | 15     | Type-Token Ratio (TTR)           |
| Filler Words        | 10     | Counts 15+ filler word patterns  |
| Sentiment           | 10     | Positivity and engagement        |
Total = 100 points

Grades:

85–100: Excellent

70–84: Good

55–69: Fair

Below 55: Needs Improvement


## 📂 Project Structure

```text
Self_Introduction_AI/
│
├── app.py                     # Main Streamlit Application logic
├── requirements.txt           # Python dependencies
├── assets/
│   └── Gemini_Generated_Image.png   # App Banner
├── README.md                  # Project Documentation
└── .streamlit/
    └── config.toml            # UI Configuration

