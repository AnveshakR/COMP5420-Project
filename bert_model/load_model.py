import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

# Load the pretrained GoEmotions model
model = TFBertForSequenceClassification.from_pretrained("monologg/bert-base-cased-goemotions-original", from_pt=True)
tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")


# csv paths 
input_csv_path = "test_input.csv" # Change this
output_csv_path = "test_output.csv" # Change this

# Read input data from CSV
input_data = pd.read_csv(input_csv_path, encoding='latin1')

# Create list to store predicted emotions
predicted_emotions = []

count = 0

# Perform emotion classification for each text 
for text in input_data["text"]:
    inputs = tokenizer([str(text)], return_tensors="tf", max_length=512, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    # Most Likely Emotion
    predicted_class = tf.argmax(logits, axis=1).numpy()[0] 
    predicted_emotions.append(predicted_class)

    # Uncomment this and comment 2 lines above for second emotion
    #second_most_probable_class = tf.argsort(logits, axis=1, direction='DESCENDING').numpy()[0, 1]
    #predicted_emotions.append(second_most_probable_class)

    count = count + 1
    if ( count % 100 == 0): # Just here to check if it running properly
        print(count)

# Add the predicted emotions to the input_data DataFrame
input_data["predicted_emotion"] = predicted_emotions

# Save the DataFrame to a new CSV file
input_data.to_csv(output_csv_path, index=False)

print("Output CSV has been created!")