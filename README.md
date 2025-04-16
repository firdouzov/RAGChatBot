# Conversational Retrieval Chatbot

A Streamlit-based chatbot that combines document retrieval with language generation capabilities. This chatbot can translate input, retrieve relevant documents from a corpus, and generate contextual responses.

## Features

- **Multi-language Support**: Automatically translates user queries using Google Translator
- **Semantic Search**: Retrieves relevant documents using sentence embeddings and FAISS
- **Conversational Memory**: Maintains conversation history for context-aware responses
- **LLM-Powered Responses**: Generates human-like responses using Llama 3.1 (8B parameter model)
- **Interactive Web Interface**: Built with Streamlit for easy interaction

## Technical Architecture

The chatbot consists of three main components:

1. **Retriever**: Uses `SentenceTransformer` with the 'all-MiniLM-L6-v2' model to create embeddings and FAISS for efficient similarity search
2. **Generator**: Implements Meta's Llama 3.1 (8B) model with 4-bit quantization for memory efficiency
3. **Translator**: Incorporates Google Translator for handling non-English queries

## Requirements

- Python 3.8+
- PyTorch
- CUDA-compatible GPU (recommended)
- Hugging Face account for model access

## Installation

```bash
# Clone the repository
git clone https://github.com/firdouzov/conversational-retrieval-chatbot.git
cd conversational-retrieval-chatbot

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Log in to Hugging Face (required to access Llama 3.1)
huggingface-cli login
```

## Configuration

Create a `requirements.txt` file with the following dependencies:

```
streamlit
faiss-cpu  # Use faiss-gpu if using GPU
torch
sentence-transformers
transformers
deep-translator
huggingface-hub
accelerate
bitsandbytes
```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

2. Access the app in your browser (typically at http://localhost:8501)

3. Enter your corpus in the sidebar (one document per line)

4. Start chatting in the main interface

## Customization

### Changing the Retrieval Model

To use a different sentence embedding model, modify:

```python
self.retriever_model = SentenceTransformer('your-preferred-model')
```

### Adjusting Generation Parameters

Modify the generation settings in the `generate_answer` method:

```python
outputs = self.generator_model.generate(
    **inputs,
    max_length=100,         # Adjust maximum response length
    temperature=0.5,        # Control randomness (higher = more random)
    top_p=0.9,              # Nucleus sampling parameter
    num_return_sequences=1, # Number of response candidates
    do_sample=False         # Set to True for more varied responses
)
```

### Translation Languages

Change the default translation languages:

```python
def translate_text(self, text, source_lang='your-source-lang', target_lang='your-target-lang'):
```

## Limitations

- Requires significant GPU memory for the Llama 3.1 model, even with quantization
- Performance may vary based on the quality and relevance of the provided corpus
- Translation quality depends on Google Translator's capabilities


## Acknowledgments

- [Hugging Face](https://huggingface.co/) for transformer models
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [Sentence Transformers](https://www.sbert.net/) for text embeddings
- [Streamlit](https://streamlit.io/) for the web interface
- [Meta AI](https://ai.meta.com/) for the Llama 3.1 model


Note, I have writing readme
