# Korean Text Summarization

This project implements a Korean text summarization model using TF-IDF and ROUGE metrics for evaluation. The system processes a collection of Korean documents, extracts key sentences, and evaluates the generated summaries against reference summaries. 

## Table of Contents

- [Project Overview](#project-overview)
- [Data Preparation](#data-preparation)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Training](#training)
  - [Summarization and Evaluation](#summarization-and-evaluation)
- [Evaluation](#evaluation)
- [License](#license)

## Project Overview

This project is designed to summarize Korean text by identifying key sentences using TF-IDF-based vectorization. Summaries are evaluated using ROUGE metrics to assess the quality of generated summaries against reference summaries.

## Data Preparation

Place the data files in the `datas` directory. Each file should be in JSON format with the following structure:

```json
{
  "content": "The main text of the document goes here.",
  "summary": "The reference summary goes here."
}
```

**Requirements for the dataset:**
- Each JSON file should contain both `content` and `summary` fields.
- Files should have a minimum size of 1KB for processing efficiency.

**Stopwords:**  
A file named `stopwords-ko.txt` should be included in the project directory, containing Korean stopwords (one per line) to filter out common words during text processing.

## Setup and Installation

1. Clone the repository and navigate to the project directory.
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the tokenizer:

   ```python
   from transformers import ElectraTokenizer
   ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
   ```

## Usage

### Training

Run the following command to load the data, preprocess the text, and train the TF-IDF model:

```python
python main.py
```

This script will:
- Load the JSON files in `datas` directory.
- Clean and preprocess the text.
- Train a TF-IDF vectorizer on the processed text.
- Save the trained model to `model.pkl`.

### Summarization and Evaluation

The script will then generate summaries using the trained TF-IDF model. Summaries are created by selecting the top sentences based on cosine similarity to the average document vector. 

After generating summaries, the script evaluates them against reference summaries using ROUGE metrics.

## Evaluation

ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L) are used to measure the similarity between the generated and reference summaries. The scores are displayed as averages for all documents processed.

```plaintext
ROUGE Scores:
{
  "ROUGE-1": { "f": 0.50, "p": 0.52, "r": 0.49 },
  "ROUGE-2": { "f": 0.30, "p": 0.31, "r": 0.29 },
  "ROUGE-L": { "f": 0.47, "p": 0.48, "r": 0.46 }
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.