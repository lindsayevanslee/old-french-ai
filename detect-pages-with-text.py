import os
from my_secrets import replicate_key
import replicate


#Set the input directory
input_dir = "data/ME MSS Images/test images"

# Set environment variable
os.environ["REPLICATE_API_TOKEN"] = replicate_key

#Model question
question = "What is this a picture of?"


def detect_pages_with_text(directory):
  for root, dirs, files in os.walk(directory):
    for file in files:
      image_path = os.path.join(root, file)

      #print the image path
      print(f"Running model for: {image_path}")

      #Set the input for the model
      model_input = {
        "task": "visual_question_answering",
        "image": open(image_path, "rb"),
        "question": question
      }

      #Run the model
      # Found this model here: https://replicate.com/salesforce/blip
      output = replicate.run(
        "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
        input=model_input
      )

      # remove "Answer: " from the output
      print(f"The answer is: {output.replace('Answer: ', '')}")


#Run the function
detect_pages_with_text(input_dir)
                







