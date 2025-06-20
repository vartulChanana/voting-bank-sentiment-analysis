# 🗳️ Voting Bank Analysis using Sentiment Analysis

This project analyzes textual feedback using Natural Language Processing (NLP) techniques to determine both **sentiment** and **dominant emotions** (like happy, sad, angry, etc.) from user inputs. It is designed to help interpret public opinion from voter statements — whether manually entered or provided in bulk through an Excel file.

---

## 📌 Features

- ✅ Analyze individual sentences for:
  - Overall **Sentiment** (Positive / Neutral / Negative)
  - **Dominant Sub-Emotion** (Happy, Sad, Angry, Fear, Surprised, Disgust)
- 📁 Upload an Excel file containing multiple statements for bulk analysis
- 📊 Automatically generates a new Excel file with sentiment & emotion scores
- 🔍 Handles **intensifiers**, **negations**, **political phrases**, and **emotion-specific keywords**
- 🧠 Uses:
  - NLTK’s **VADER Sentiment Analyzer**
  - Custom-built keyword and phrase-based emotion detection

--- 

HOW TO USE 

## 🖼️ Demo

```bash
Option 1: Enter text manually
> "I can't believe this happened. It's just so unfair!"
Sentiment: Negative
Dominant Emotion: Angry

Option 2: Analyze Excel file
> Upload `us_election_2024.xlsx` → Output: `analyzed_us_election_2024.xlsx

