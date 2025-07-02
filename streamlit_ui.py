import streamlit as st
from PIL import Image
from transformers import pipeline
from PIL import Image

st.title("Image Question Answering Chatbot")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image.save('..//demo.jpg')

Question = st.text_input("Enter your question you wants to ask about the image:")

if Question and image is not None:
    # Load image
    image = Image.open("demo.jpg").convert("RGB")

    # Load VQA pipeline (downloads model automatically)
    vqa = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")

    # Ask question
    result = vqa(image, Question)
    answer=result[0]["answer"]
    # Print answer
    st.write("Answer:", answer)




