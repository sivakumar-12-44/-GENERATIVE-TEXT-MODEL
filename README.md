# -GENERATIVE-TEXT-MODEL

COMPANY: CODTECH IT SOLUTIONS

NAME: BOLLAVARAM SIVA KUMAR REDDY

INTERN ID: CT04WT242

DOMAIN: ARTIFICIAL INTELLIGENCE

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

#This Python script demonstrates two distinct approaches to text generation using deep learning—one using GPT-2 from the Hugging Face Transformers library, and the other using a custom LSTM-based neural network built with TensorFlow/Keras. In the first part, the function generate_text_gpt2() loads the pre-trained GPT-2 model and tokenizer, takes a user-defined prompt, and generates text by sampling from the model's output distribution using parameters like top_p, temperature, and no_repeat_ngram_size to control creativity and coherence. It outputs fluent, human-like text based on the prompt, showcasing the power of large transformer-based language models.

The second part builds a simpler, character-level LSTM model for next-word prediction. It starts by defining a small corpus of meaningful sentences and tokenizing the text into sequences using Keras' Tokenizer. It then prepares input sequences using n-grams and pads them to a uniform length. The model architecture includes an embedding layer, an LSTM layer, and a dense output layer with a softmax activation to predict the next word. After training the model for 300 epochs, the function generate_text_lstm() takes a seed phrase and generates new text word-by-word, predicting one word at a time based on the learned patterns in the corpus. This script effectively contrasts a state-of-the-art transformer-based approach with a classic RNN-based method, illustrating the evolution and capabilities of modern natural language generation techniques.

OUTPUT:

![Image](https://github.com/user-attachments/assets/82f04faf-6243-4eea-b7ef-b878c2d273ac)
