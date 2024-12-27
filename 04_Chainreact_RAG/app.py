import streamlit as st
from streamlit_option_menu import option_menu
import openai
import numpy as np
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import warnings
import os

warnings.filterwarnings("ignore")

# Sidebar for navigation and API key input
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
openai.api_key = api_key

with st.sidebar:
    page = option_menu(
        "Dashboard",
        ["Home", "About Me", "Chain React"],
        icons=['house', 'info-circle',  'file-text'],
        menu_icon="list",
        default_index=0,
    )

if not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to use the application.")

else:
    if page == "Home":
        st.title("Welcome to ChainReact!")
        st.write("ChainReact is an interactive program designed to help professionals in logistics, supply chain management, and data analysis optimize and streamline their operations. Whether you’re managing inventory, overseeing production, or ensuring timely order fulfillment, ChainReact provides actionable insights and data-driven solutions that make managing the complexities of the supply chain easier and more efficient.")

        st.write("## How It Works")
        st.write("ChainReact uses real-time data to simulate and optimize key supply chain components such as sourcing, production, inventory management, transportation, and order fulfillment. The program integrates industry best practices and AI-driven suggestions to help users identify inefficiencies, reduce costs, and improve overall supply chain performance. With intuitive interfaces, users can easily input and analyze their data, visualize workflows, and implement strategies for continuous improvement.")

        st.write("## Ideal Users")
        st.write("**- Supply Chain Managers** looking for efficient ways to streamline logistics and production processes.")
        st.write("**- Logistics Coordinators** seeking real-time solutions for transportation and distribution optimization.")
        st.write("**- Data Analysts** interested in using data-driven insights to improve inventory control, production planning, and order management.")
        st.write("**- Small to Medium Enterprises (SMEs)** looking to scale their supply chain operations with optimized strategies.")

        st.write("Whether you’re managing a global network or a local supply chain, ChainReact adapts to your needs and helps you make smarter decisions faster.")

    elif page == "About Me":
        st.header("About Me")
        st.markdown("""
         Hi! I'm Andrea Sombilon! I am a business intelligence specialist and data analyst with a strong foundation in deriving insights from data to drive strategic decisions. Currently, I am expanding my skill set by learning to create products in Artificial Intelligence and working towards becoming an AI Engineer. My goal is to leverage my expertise in BI and data analysis with advanced AI techniques to create innovative solutions that enhance business intelligence and decision-making capabilities. 
        
        This projects is one of the projects I am building to try and apply the learning I have acquired from the AI First Bootcamp of AI Republic.
        
        Any feedback would be greatly appreciated! ❤
        """)
        
        st.text("Connect with me on LinkedIn 😊 [Andrea Arana](https://www.linkedin.com/in/andrea-a-732769168/)")

    elif page == "Chain React":
        dataframed = pd.read_csv('https://raw.githubusercontent.com/11andrea2233/ChainReact/refs/heads/main/Transportation%20and%20distribution.csv')
        dataframed['combined'] = dataframed.apply(lambda row : ' '.join(row.values.astype(str)), axis = 1)
        documents = dataframed['combined'].tolist()
        embeddings = [get_embedding(doc, engine = "text-embedding-3-small") for doc in documents]
        embedding_dim = len(embeddings[0])
        embeddings_np = np.array(embeddings).astype('float32')
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings_np)
        
        System_Prompt = """
            Role: Data Analyst
            Intent: I want to analyze the transportation data to optimize route planning and reduce costs.
            Context: I have a dataset containing shipment details including the mode of transport, origin, destination, transit time, and costs. The dataset is stored in a CSV file titled "Transportation and distribution.csv".
            Constraint: The dataset only includes data for a few selected routes and modes of transportation, limiting the ability to generalize findings across all possible transportation scenarios. Costs are provided in different formats, requiring normalization for accurate analysis.
            Examples:
                Example1: Using this dataset, can you calculate the average transit time and cost for shipments made by truck, and identify which truck route is the most expensive on average?
"""

        def initialize_conversation(prompt):
            if 'message' not in st.session_state:
                st.session_state.message = []
                st.session_state.message.append({"role": "system", "content": System_Prompt})
                chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.message, temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
                response = chat.choices[0].message.content
                st.session_state.message.append({"role": "assistant", "content": response})

        initialize_conversation(System_Prompt)

        for messages in st.session_state.message:
            if messages['role'] == 'system':
                continue
            else:
                with st.chat_message(messages["role"]):
                    st.markdown(messages["content"])

        if user_message := st.chat_input("Ask me anything about Supply Chain and Logistics!"):
            with st.chat_message("user"):
                st.markdown(user_message)
            query_embedding = get_embedding(user_message, engine='text-embedding-3-small')
            query_embedding_np = np.array([query_embedding]).astype('float32')    
            _, indices = index.search(query_embedding_np, 2)
            retrieved_docs = [documents[i] for i in indices[0]]
            context = ' '.join(retrieved_docs)
            structured_prompt = f"Context:\n{context}\n\nQuery:\n{user_message}\n\nResponse:"
            chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.message + [{"role": "user", "content": structured_prompt}], temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
            st.session_state.message.append({"role": "user", "content": user_message})
            chat = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=st.session_state.message,
            )
            response = chat.choices[0].message.content
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.message.append({"role": "assistant", "content": response})