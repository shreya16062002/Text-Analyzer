import streamlit as st
import spacy
from spacy import displacy
from io import StringIO
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
nlp = spacy.load("en_core_web_sm")

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')

def tokenize(text):
    doc = nlp(text)
    tokens = []

    for token in doc:
        tokens.append({
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_
        })

    return tokens


def syntactic_analysis(text):
    doc = nlp(text)
    result = [[token.text, token.pos_, spacy.explain(token.pos_)] for token in doc]
    return result



def sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity

    # Categorize sentiment
    sentiment_label = "Positive" if sentiment_polarity > 0 else "Negative" if sentiment_polarity < 0 else "Neutral"

    # Explanation of subjectivity
    subjectivity_explanation = (
        "Very Objective" if sentiment_subjectivity < 0.3
        else "Objective" if sentiment_subjectivity < 0.6
        else "Subjective" if sentiment_subjectivity < 0.8
        else "Very Subjective"
    )

    return {
        "sentiment": sentiment_label,
        "subjectivity": subjectivity_explanation
    }


def summarization(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary_sentences = summarizer(parser.document, num_sentences)

    # Combine the summary sentences into a paragraph
    summary_paragraph = ' '.join(str(sentence) for sentence in summary_sentences)

    return summary_paragraph

# Streamlit UI
st.title("Text Analysis Web App")

text_input = st.text_area("Enter text here:")

if st.button("Tokenize"):
    tokens = tokenize(text_input)
    st.write(tokens)

if st.button("Syntactic Analysis"):
    syntactic_result = syntactic_analysis(text_input)
    for idx, item in enumerate(syntactic_result):
        st.write(f"{idx}: [{item[0]}, {item[1]} ({item[2]})]")


if st.button("Sentiment Analysis"):
    sentiment_result = sentiment_analysis(text_input)
    st.write("Sentiment:", sentiment_result["sentiment"])
    st.write("Subjectivity:", sentiment_result["subjectivity"])

    # Count and explanation
    if sentiment_result["sentiment"] == "Positive":
        st.write("Positive Count: 1")
        st.write("Explanation: The text expresses positive sentiment.")
    elif sentiment_result["sentiment"] == "Negative":
        st.write("Negative Count: 1")
        st.write("Explanation: The text expresses negative sentiment.")
    else:
        st.write("Neutral Count: 1")
        st.write("Explanation: The text is neutral or lacks a clear sentiment.")

if st.button("Summarization"):
    summary = summarization(text_input)
    st.write("Summary:", summary)