from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_text_gpt2(prompt, max_length=150):
    model_name = 'gpt2'  # You can use 'gpt2-medium' or 'gpt2-large' for better results
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.9,
            top_p=0.95,
            do_sample=True
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


prompt = "The future of artificial intelligence"
print("🧠 GPT-2 Generated Text:\n")
print(generate_text_gpt2(prompt))


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


corpus = [
    "Climate change is one of the biggest challenges facing the world.",
    "Artificial intelligence is transforming every aspect of our lives.",
    "Space exploration pushes the boundaries of human achievement.",
    "Renewable energy is essential for a sustainable future.",
    "Education is the key to unlocking human potential."
]


tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1


input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram = token_list[:i+1]
        input_sequences.append(n_gram)


max_seq_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = np.eye(total_words)[y]


model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_seq_len-1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=300, verbose=0)

def generate_text_lstm(seed_text, next_words=20):
    result = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([result])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                result += ' ' + word
                break
    return result


seed = "Education is"
print("\n📚 LSTM Generated Text:\n")
print(generate_text_lstm(seed))
