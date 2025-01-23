import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from deep_translator import GoogleTranslator

class ConversationalRetrievalChatbot:
    def __init__(self, corpus):
        self.corpus = corpus
        self.conversation_history = []
        
        # Initialize models
        self.retriever_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.generator_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.generator_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        
        # Precompute corpus embeddings
        self.corpus_embeddings = self.retriever_model.encode(corpus, convert_to_tensor=True).cpu().numpy()
        faiss.normalize_L2(self.corpus_embeddings)
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.corpus_embeddings.shape[1])
        self.index.add(self.corpus_embeddings)
    
    def translate_text(self, text, source_lang='az', target_lang='en'):
        # Words to exclude from translation
        exclude_words = ["besde bes"]
        placeholders = {word: f"PLACEHOLDER_{i}" for i, word in enumerate(exclude_words)}
        
        # Replace excluded words with placeholders
        for word, placeholder in placeholders.items():
            text = text.replace(word, placeholder)
        
        # Translate the text
        translated_text = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        
        # Replace placeholders back with original words
        for word, placeholder in placeholders.items():
            translated_text = translated_text.replace(placeholder, word)
        
        return translated_text
    
    def retrieve(self, query, top_k=1):
        # Encode the query
        query_embedding = self.retriever_model.encode([query], convert_to_tensor=True)
        query_embedding_np = query_embedding.cpu().numpy()
        faiss.normalize_L2(query_embedding_np)
        
        # Perform the search
        D, I = self.index.search(query_embedding_np, top_k)
        
        # Retrieve the documents
        return [self.corpus[i] for i in I[0]]
    
    def generate_answer(self, query, retrieved_docs):
        # Combine conversation history with retrieved docs
        context = " ".join(self.conversation_history + retrieved_docs)
        input_text = f"question: {query} context: {context}"
        
        # Generate answer
        inputs = self.generator_tokenizer(input_text, return_tensors='pt')
        outputs = self.generator_model.generate(inputs['input_ids'], max_length=50)
        answer = self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Update conversation history
        self.conversation_history.append(query)
        self.conversation_history.append(answer)
        
        return answer
    
    def chat(self, query):
        # Translate query
        translated_query = self.translate_text(query)
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(translated_query)
        filtered_docs = [doc for doc in retrieved_docs if len(doc.split()) > 5]
        
        # Generate and return answer
        return self.generate_answer(translated_query, filtered_docs)

def main():
    st.title("Conversational Retrieval Chatbot")
    
    # Corpus input
    st.sidebar.header("Corpus Input")
    corpus_input = st.sidebar.text_area("Enter Corpus (one document per line)")
    
    # Initialize chatbot when corpus is provided
    if corpus_input:
        corpus = [doc.strip() for doc in corpus_input.split('\n') if doc.strip()]
        chatbot = ConversationalRetrievalChatbot(corpus)
        
        # Initialize chat history in session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question"):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate bot response
            with st.chat_message("assistant"):
                response = chatbot.chat(prompt)
                st.markdown(response)
            
            # Add bot response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("Please enter a corpus in the sidebar to start chatting.")

if __name__ == "__main__":
    main()