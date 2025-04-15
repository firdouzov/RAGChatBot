import streamlit as st
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from deep_translator import GoogleTranslator
from huggingface_hub import login
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)
with init_empty_weights():
    generator_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B", 
        device_map="cuda", 
        torch_dtype=torch.float16,
        quantization_config=quantization_config
    )

generator_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

class ConversationalRetrievalChatbot:
    def __init__(self, corpus,generator_model,generator_tokenizer):
        # Clear GPU cache
        torch.cuda.empty_cache()

        self.corpus = corpus
        self.conversation_history = []


        # Embedding Model
        self.retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.generator_model = generator_model
        self.generator_tokenizer = generator_tokenizer


        # Precompute normalized corpus embeddings
        self.corpus_embeddings = self.retriever_model.encode(
            corpus, 
            convert_to_tensor=True, 
            normalize_embeddings=True
        ).cpu().numpy()

        # FAISS index
        self.index = faiss.IndexFlatIP(self.corpus_embeddings.shape[1])
        self.index.add(self.corpus_embeddings)

    def translate_text(self, text, source_lang='az', target_lang='en'):
        exclude_words = ["besde bes"]
        placeholders = {word: f"PLACEHOLDER_{i}" for i, word in enumerate(exclude_words)}

        for word, placeholder in placeholders.items():
            text = text.replace(word, placeholder)

        translated_text = GoogleTranslator(source=source_lang, target=target_lang).translate(text)

        for word, placeholder in placeholders.items():
            translated_text = translated_text.replace(placeholder, word)

        return translated_text

    def retrieve(self, query, top_k=1):
        query_embedding = self.retriever_model.encode(
            [query], 
            normalize_embeddings=True, 
            convert_to_tensor=True
        ).cpu().numpy()

        D, I = self.index.search(query_embedding, top_k)
        return [self.corpus[i] for i in I[0] if i < len(self.corpus)]

    def generate_answer(self, query, retrieved_docs):
        context = " ".join(self.conversation_history + retrieved_docs[:1])
        input_text = f"Question: {query}\nContext: {context}\nAnswer:"

        inputs = self.generator_tokenizer(
            input_text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512
        ).to("cuda")

        outputs = self.generator_model.generate(
            **inputs,
            max_length=100,
            temperature=0.5,
            top_p=0.9,
            num_return_sequences=1,
            do_sample=False
        )

        answer = self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

        self.conversation_history.append(query)
        self.conversation_history.append(answer)

        return answer

    def chat(self, query):
        translated_query = self.translate_text(query)
        retrieved_docs = self.retrieve(translated_query, top_k=1)
        filtered_docs = [doc for doc in retrieved_docs if len(doc.split()) > 5]
        return self.generate_answer(translated_query, filtered_docs)

def main():
    st.title("Conversational Retrieval Chatbot")

    st.sidebar.header("Corpus Input")
    corpus_input = st.sidebar.text_area("Enter Corpus (one document per line)")

    if corpus_input:
        corpus = [doc.strip() for doc in corpus_input.split('\n') if doc.strip()]
        chatbot = ConversationalRetrievalChatbot(
            corpus, 
            generator_model, 
            generator_tokenizer
        )

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question"):
            with st.chat_message("user"):
                st.markdown(prompt)

            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                response = chatbot.chat(prompt)
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("Please enter a corpus in the sidebar to start chatting.")

if __name__ == "__main__":
    main()
