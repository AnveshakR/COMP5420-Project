import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

# Load the pretrained GoEmotions model
model = TFBertForSequenceClassification.from_pretrained("monologg/bert-base-cased-goemotions-original", from_pt=True)
tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")


# csv paths
input_csv_path = "test_input.csv"
output_csv_path = "test_output.csv"

# Read input data from CSV
input_data = pd.read_csv(input_csv_path, encoding='latin1')

# Create list to store predicted emotions
predicted_emotions = []

# Perform emotion classification for each text 
for text in input_data["text"]:
    inputs = tokenizer(text, return_tensors="tf", max_length=512, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    predicted_emotions.append(predicted_class)

# Add the predicted emotions to the input_data DataFrame
input_data["predicted_emotion"] = predicted_emotions

# Save the DataFrame to a new CSV file
input_data.to_csv(output_csv_path, index=False)

print("Output CSV has been created!")