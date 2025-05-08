import gradio as gr
from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    label = result['label']
    confidence = result['score']
    return f"Sentiment: {label}, Confidence: {confidence:.2f}"

interface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter text here..."),
    outputs="text",
    title="Sentiment Analysis",
    description="Classify sentiment of the input text (positive, negative, or neutral)."
)

interface.launch()
