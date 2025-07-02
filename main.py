from transformers import pipeline
from PIL import Image

# Load image
image = Image.open("demo.jpg").convert("RGB")

# Load VQA pipeline (downloads model automatically)
vqa = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")

# Ask question
question = "how many boys are in image"
result = vqa(image, question)

# Print answer
print("Answer:", result[0]["answer"])
