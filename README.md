# Ngram Speller 🧠📝

A simple yet effective spelling correction tool using **n-gram language models**. This project was developed as a part of a Machine Learning mini-project and demonstrates the power of probabilistic language models in handling spelling errors and suggesting corrections.

## 🔍 Overview

The Ngram Speller takes noisy (misspelled) text as input and attempts to correct it using statistical n-gram analysis trained on a corpus of text. It uses ngram probabilities and minimum edit distance to rank possible corrections, helping identify the most likely intended sentence.

## ✨ Features

- Customizable n-gram model (unigram, bigram, trigram)
- Edit distance-based candidate generation
- Probability-based scoring to select the best correction
- Easy to use and extend
- Pure Python implementation

## 📁 Project Structure

ngram_speller/ <br>
├── data/ <br>
│ └── .... # Data for test and train <br>
├── models/ <br>
│ └── min_edit_dist.py # Implementation of the min edit distance algorithm <br>
│ └── ngram_model.py # Ngram language model implementation <br>
├── main.py <br>
│ └── speller.py # Main spelling correction logic <br>
├── README.md # Project documentation<br>
└── .... # **uv** package manager files <br>

## 🛠 How It Works

1. Training: The model reads a corpus and builds ngram and context (n-1 gram) frequency tables.
2. Candidate Generation: For each misspelled word, the model generates candidates using context.
3. Scoring: Each candidate is scored on its minimum distance with the mispelled word.
4. Correction: The best candidate (based on score) replaces the misspelled word.

## 🚀 Testing the project

### Using uv:
```bash
# clone the repo
git clone https://github.com/hamza-rx12/ngram_speller.git    
# cd into the project directory
cd ngram_speller
# set up the virtual environment   
uv venv
# install the requirements   
uv install
# run the main.py file   
uv run    
```
