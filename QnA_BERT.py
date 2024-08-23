import pandas as pd
import numpy as np
import torch
from transformers import BertForQuestionAnswering, BertTokenizer

# Load and process the dataset
qa_dataset = pd.read_json('http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json')
del qa_dataset["version"]

# Required columns for our dataframe
columns_needed = ["context", "query", "response"]
data_list = []

# Create a new DataFrame with the required columns
for idx, entry in qa_dataset.iterrows():
    for i in range(len(entry["data"]["questions"])):
        row = [
            entry["data"]["story"],
            entry["data"]["questions"][i]["input_text"],
            entry["data"]["answers"][i]["input_text"]
        ]
        data_list.append(row)

qa_df = pd.DataFrame(data_list, columns=columns_needed)
qa_df.to_csv("processed_qa_data.csv", index=False)

# Load processed data
qa_data = pd.read_csv("processed_qa_data.csv")
qa_data.head()
print("Number of Q&A pairs: ", len(qa_data))

# Load BERT model and tokenizer
qa_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
qa_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Randomly select a question and context pair
random_idx = np.random.randint(0, len(qa_data))
query = qa_data["query"][random_idx]
context = qa_data["context"][random_idx]

# Tokenize input
input_ids = qa_tokenizer.encode(query, context)
print("The input has a total of {} tokens.".format(len(input_ids)))

tokens = qa_tokenizer.convert_ids_to_tokens(input_ids)
for token, id in zip(tokens, input_ids):
    print('{:8}{:8,}'.format(token, id))

# Segment IDs
sep_idx = input_ids.index(qa_tokenizer.sep_token_id)
print("SEP token index: ", sep_idx)
num_seg_a = sep_idx + 1
print("Number of tokens in segment A: ", num_seg_a)
num_seg_b = len(input_ids) - num_seg_a
print("Number of tokens in segment B: ", num_seg_b)
segment_ids = [0] * num_seg_a + [1] * num_seg_b
assert len(segment_ids) == len(input_ids)

# Run the model
outputs = qa_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

# Get tokens with the highest start and end scores
answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits)
if answer_end >= answer_start:
    answer = " ".join(tokens[answer_start:answer_end + 1])
else:
    print("Unable to find the answer to this question. Please ask another question.")

print("\nQuery:\n{}".format(query.capitalize()))
print("\nResponse:\n{}.".format(answer.capitalize()))

# Post-process the answer
answer = tokens[answer_start]
for i in range(answer_start + 1, answer_end + 1):
    if tokens[i].startswith("##"):
        answer += tokens[i][2:]
    else:
        answer += " " + tokens[i]

def get_answer(query, context):
    # Tokenize the question and context as a pair
    input_ids = qa_tokenizer.encode(query, context)

    # Convert tokenized IDs to tokens
    tokens = qa_tokenizer.convert_ids_to_tokens(input_ids)

    # Segment IDs
    sep_idx = input_ids.index(qa_tokenizer.sep_token_id)
    num_seg_a = sep_idx + 1
    num_seg_b = len(input_ids) - num_seg_a

    segment_ids = [0] * num_seg_a + [1] * num_seg_b
    assert len(segment_ids) == len(input_ids)

    # Run the model to get outputs
    outputs = qa_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits)
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start + 1, answer_end + 1):
            if tokens[i].startswith("##"):
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]

    if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."

    print("\nPredicted answer:\n{}".format(answer.capitalize()))

# User interaction
context = input("Please enter the context: \n")
query = input("\nPlease enter your question: \n")
while True:
    get_answer(query, context)

    continue_flag = True

    while continue_flag:
        response = input("\nDo you want to ask another question based on this context (Y/N)? ")
        if response.lower().startswith("y"):
            query = input("\nPlease enter your question: \n")
            continue_flag = False
        elif response.lower().startswith("n"):
            print("\nGoodbye!")
            continue_flag = False
            break
    if not continue_flag:
        break
