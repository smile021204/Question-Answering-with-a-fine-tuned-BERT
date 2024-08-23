# Question Answering System Using Fine-Tuned BERT

## Project Overview

This project implements a question-answering system using a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model. The system is capable of taking a context (a passage of text) and answering questions based on that context. It leverages pre-trained BERT models, fine-tuned on the SQuAD dataset, to achieve high accuracy in predicting the start and end positions of the answer within the text.

## Features

- **Data Processing:** The project processes the CoQA dataset to extract relevant context, questions, and answers. The dataset is then saved as a CSV file for further use.
- **Model Architecture:** The system uses a fine-tuned BERT model (`bert-large-uncased-whole-word-masking-finetuned-squad`) from the Hugging Face Transformers library to perform the question-answering task.
- **Interactive Question Answering:** The system allows users to input their own context and questions, providing real-time answers using the fine-tuned BERT model.

## How It Works

1. **Data Preprocessing:** The project begins by loading and preprocessing the CoQA dataset, extracting stories (context), questions, and answers, and then saving the processed data into a CSV file.
  
2. **Model and Tokenizer Setup:** The BERT model and tokenizer are loaded from the Hugging Face Transformers library. The model is fine-tuned specifically for question-answering tasks.

3. **Answer Extraction:** Given a context and a question, the system tokenizes the input, feeds it into the BERT model, and extracts the answer by identifying the tokens with the highest start and end logits.

4. **User Interaction:** The system allows users to input a context and ask questions interactively. The model processes the input and returns the most probable answer.

## Requirements

- Python 3.6+
- PyTorch
- Transformers (Hugging Face)
- Pandas
- NumPy

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/qa-bert-project.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:
   ```bash
   python main.py
   ```

## Data

The project uses the CoQA dataset, which can be downloaded directly from the Stanford NLP group. This dataset is then processed to extract the necessary columns for question answering.

## Source

This project is inspired by the article "[Question Answering with a Fine-Tuned BERT](https://towardsdatascience.com/question-answering-with-a-fine-tuned-bert-bc4dafd45626)" published on Towards Data Science.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
