import os
import csv
import openai
import numpy as np
import pandas as pd
import folium
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
from streamlit_folium import st_folium
from folium import plugins
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import logging

warnings.filterwarnings("ignore")

st.set_page_config(page_title="📈 StockPrize Ally", layout="wide")

System_Prompt_Forecast = """
Role:
You are StockPrize Ally, an AI-based Stock Price Forecasting Model designed to generate predictions of future stock prices based on historical data. Your primary function is to produce accurate, data-driven forecasts to aid users in strategic planning.

Instructions:

Accept a list of historical stock prices data as input, consisting of numerical values representing closing price, open price, high price, low price, and volume for past periods.
Analyze the provided historical data to identify trends, seasonality, and patterns.
Generate a stock prices forecast for the next 12 periods using appropriate statistical or machine learning models.
Output the forecasted values as a comma-separated string for easy parsing and into a line chart with the months as x and stock prices as y. 
Ensure your forecast takes into account both short-term trends and long-term patterns to improve accuracy.
Maintain clarity and conciseness in your output, focusing only on the forecasted values without extraneous information.

Context:
The user will input a series of numerical values representing closing price, open price, high price, low price, and volume over a sequence of past periods (e.g., monthly closing price, open price, high price, low price, and volume for the past two years). Your task is to predict the stock prices for the next 12 periods based on this historical data. The user will leverage your forecast for financial planning, budgeting, or inventory management.

Constraints:

Do not assume any additional data beyond what the user provides (e.g., macroeconomic factors or market conditions).
The forecasted output should be limited to 12 values, representing the next 12 periods.

Examples:

Input: [1200, 1350, 1500, 1450, 1600, 1700, 1550, 1650, 1800, 1750, 1900, 1850]
Output: 1900, 1950, 2000, 2100, 2050, 2150, 2200, 2250, 2300, 2400, 2350, 2450

Input: [100, 200, 300, 250, 350, 400, 450, 500, 550, 600, 650, 700]
Output: 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300
"""
System_Prompt_Explanation = """
You are StockPrize Ally, AI-based Stock Price Forecast Explanation Model designed to provide clear, insightful interpretations of the forecasted values generated by the forecasting model. Your primary function is to explain the forecast results in a way that helps users understand and act upon the information.

Instructions:

Analyze the forecasted Stock Price values and identify significant trends, such as growth patterns, seasonality, or unexpected fluctuations.
Interpret what the forecasted values imply about future business performance, focusing on areas like stock prices growth, potential slowdowns, or cyclical changes.
Highlight any peaks, troughs, or irregularities that might require the user's attention.
Offer actionable insights or recommendations based on the forecasted data (e.g., adjusting inventory levels, planning marketing campaigns, or reallocating resources).
Ensure explanations are clear, concise, and tailored to the user’s needs, focusing on helping them make strategic decisions.

Context:
The forecasted closing price, open price, high price, low price, and volume data you receive will be based on historical closing price, open price, high price, low price, and volume trends provided by the user. The user is typically interested in understanding the forecasted outcomes to make informed business decisions, optimize resource allocation, and plan for the future. Your explanations will guide the user in interpreting the forecast’s implications.

Constraints:

Do not re-run or modify the forecast calculations—focus solely on interpreting the given data.
Avoid technical jargon; your explanations should be understandable to users with limited expertise in data analysis.
Ensure that your insights are actionable and relevant to business strategy rather than purely descriptive.

Examples:

Forecasted Values: 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750
Explanation:

The forecast shows a steady upward trend, suggesting consistent growth in closing price, open price, high price, low price, and volume. This could indicate increased demand or successful closing price, open price, high price, low price, and volume strategies.
Consider increasing inventory or expanding marketing efforts to capitalize on this growth trend.
Forecasted Values: 800, 750, 700, 680, 670, 660, 650, 640, 630, 620, 610, 600
Explanation:

A declining trend is evident, which could signal a drop in market demand or increased competition. This suggests a need to review closing price, open price, high price, low price, and volume strategies or explore new Stock Price streams.
Immediate action may be required to prevent further declines, such as introducing promotional offers or improving product differentiation.
"""

# Sidebar for API key and options
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input('Enter OpenAI API token:', type='password')
    
    # Check if the API key is valid
    if api_key and api_key.startswith('sk-'):  # Removed length check
        openai.api_key = api_key
        st.success('API key is valid. Proceed to enter your closing price, open price, high price, low price, and volume data!', icon='👉')
    else:
        st.warning('Please enter a valid OpenAI API token!', icon='⚠️')

    st.header("Instructions")
    st.write("1. Enter a valid OpenAI API Key.")
    st.write("2. Click StockPrize AI on the Sidebar to get started!")
    st.write("3. Input your data.")
    st.write("4. Click 'Forecast Stock Price' to see the predictions.")

    if st.button("Reset"):
        st.session_state.clear()  # Clear session state to reset the app

    options = option_menu(
        "Content",
        ["Home", "About Me", "StockPrize AI"],
        default_index=0
    )

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Define the get_embedding function
logging.basicConfig(level=logging.INFO)


def get_embedding(document, engine="text-embedding-3-small"):
    logging.info(f"Retrieving embedding for document: {document[:30]}...")  # Log the beginning of a document to identify calls
    try:
        response = openai.Embedding.create(
            input=document,
            model=engine
        )
        return response['data']['embedding']
    except openai.error.OpenAIError as e:
        logging.error(f"Failed to retrieve embedding: {e}")
        return None


# Function to forecast stockprice
def forecast_stock_price(data, columns):
    # Prepare the input for the GPT model
    
    if isinstance(columns, list) and all(col in data.columns for col in columns):
        # Convert selected column data into a single string per row, then join all rows
        stock_price_data_str = ', '.join(data[columns].astype(str).apply(lambda row: ' '.join(row.values), axis=1))
    else:
        raise ValueError("Columns provided are not correctly specified or do not exist in the DataFrame")
   
    # RAG Implementation
    # Load and prepare data for RAG
    dataframed = pd.read_csv('https://raw.githubusercontent.com/11andrea2233/AI_Republic_Projects/refs/heads/main/05_StockPrize_Ally/HistoricalData_1726367135218.csv')
    dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    documents = dataframed['combined'].tolist()

    embeddings = [get_embedding(doc, engine="text-embedding-3-small") for doc in documents]
    embedding_dim = len(embeddings[0])
    embeddings_np = np.array(embeddings).astype('float32')

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)

    # Generate embedding for the forecast string
    query_embedding = get_embedding(stock_price_data_str, engine='text-embedding-3-small')
    query_embedding_np = np.array([query_embedding]).astype('float32')

    # Search for relevant documents
    _, indices = index.search(query_embedding_np, 2)
    retrieved_docs = [documents[i] for i in indices[0]]
    context = ' '.join(retrieved_docs)

    # Create a prompt for the GPT model
    prompt = f"""
    Given the following stock price data: {stock_price_data_str}, and the context: {context} forecast the next 12 periods of stock price. 
    Return only the forecasted values as a comma-separated string."""

    # Call the OpenAI API to generate the forecast
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        temperature= 0.1,
        messages=[
            {"role": "system", "content": System_Prompt_Forecast},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the forecasted values from the response
    forecasted_values = response['choices'][0]['message']['content']

    # Print the response for debugging
    print("API Response:", forecasted_values)

    # Convert the forecasted values to a list of floats
    try:
        forecasted_data = [float(value) for value in forecasted_values.split(',')]
    except ValueError as e:
        st.error("Error parsing forecasted values. Please check the API response.")
        print("Error:", e)
        return None

    return forecasted_data, context

# Function to generate explanation using OpenAI API
def generate_explanation(data, forecast):
    # Prepare the historical data for the prompt
    historical_data_str = data.to_string(index=False)  # Convert DataFrame to string for better readability
    forecast_str = ', '.join(map(str, forecast))  # Convert forecasted values to a string

    # Modify the prompt to focus on how the forecast was derived and analyze historical trends
    prompt = f"""
    {System_Prompt_Explanation}
    
    1. Analyze the historical revenue data provided below and identify key trends, fluctuations, and patterns:
    {historical_data_str}
    
    2. Based on the historical data, explain how the forecasted revenue values were derived: {forecast_str}.

    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        temperature= 0.7,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": "You are an AI assistant analyzing stock prices data. Provide accurate statistics and insights based on the full dataset."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message['content']

# Home Page
if options == "Home":
    st.title("Welcome to StockPrize Ally!🏆")
    st.write("""
    **Welcome to StockPrize Ally — Your Intelligent Stock Forecasting Partner**
    Navigate the complexities of the stock market with precision and confidence. StockPrize Ally harnesses the power of advanced AI to provide detailed forecasts of stock prices, including opening, closing, high, low, and volume metrics. Whether you're an investor seeking to optimize your portfolio, a financial analyst aiming for accurate market predictions, or a finance student eager to learn about market dynamics, StockPrize Ally is your go-to solution. Experience a new level of insight and control over your stock investments with real-time predictions at your fingertips.
    **Start forecasting with StockPrize Ally today and transform your approach to stock market investment.**
    """)

    st.subheader("🧿 Features")
    st.write("""
    - **User-Friendly Interface**: Navigate effortlessly through the application with a clean and intuitive design.
    - **Stock Price Forecasting**: Input your historical closing price, open price, high price, low price, and volume data, and let StockPrize Ally generate forecasts for the next 12 periods, giving you a clear view of potential future stock prices.
    - **Insightful Explanations**: Not only does StockPrize Ally provide forecasts, but it also explains how these predictions were derived, analyzing historical trends and patterns to enhance your understanding.
    - **Contextual Analysis**: The application utilizes additional contextual data to enrich the forecasting process, ensuring that your predictions are informed by relevant market insights.
    - **Automated Visualizations**: Visualize your stock prices data and forecasts with engaging charts, making it easier to grasp trends and make strategic decisions.
    """)

    st.subheader("🔑 Why StockPrize Ally?")
    st.write("""
    In an era where data is king, StockPrize Ally was created to bridge the gap between complex data analysis and actionable business insights. Whether you're a small business owner, a sales manager, or a financial analyst, having access to reliable forecasts can significantly impact your planning and strategy.
    """)

# About Us Page
elif options == "About Me":
    st.title("About Me")
    #My_image = Image.open("images/photo-me1.jpg")
    #my_resized_image = My_image.resize((400,320))
    #st.image(my_resized_image)
    st.write("I am Andrea Sombilon-Arana, an AI builder and programmer.")
    st.write("Don't hesitate to contact me on my LinkedIn and check out my other projects on GitHub! 😊")
    st.write("https://www.linkedin.com/in/andrea-a-732769168/")
    st.write("https://github.com/11andrea2233")

# Forecast Page
elif options == "StockPrize AI":
    st.title("📈 StockPrize Ally AI")
    
    # Option for user to input data
    data_input_method = st.selectbox("How would you like to input your stock prices data?", ["Upload CSV", "Enter Data Manually"])

    if data_input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your stock price data CSV", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Data Preview:", data.head())
            # Create a dropdown for selecting the column to forecast
            closing_price_column = st.selectbox("Select closing price column:", data.columns)
            opening_price_column = st.selectbox("Select opening price column:", data.columns)
            high_price_column = st.selectbox("Select high price column:", data.columns)
            low_price_column = st.selectbox("Select low price column:", data.columns)
            volume_column = st.selectbox("Select volume column:", data.columns)
            
            # Handling the column selections and checking their existence in the DataFrame
            column_names = [closing_price_column, opening_price_column, high_price_column, low_price_column, volume_column]

            # Ensure all selected columns exist in the DataFrame
            if all(col in data.columns for col in column_names):
                # Button to trigger forecasting
                if st.button("Forecast Stock Prices"):
                    forecast, context = forecast_stock_price(data, column_names)
                    st.write("Forecasted Stock Prices:", forecast)

                    explanation = generate_explanation(data, forecast)  # Assuming this function is defined to use data and forecast
                    st.write("Explanation:", explanation)
            else:
                st.error("One or more selected columns do not exist in the uploaded data. Please check your selections.")
    
    else:
        # Manual data entry
        st.write("Enter your stock prices data below:")
        closing_data = st.text_area("Closing Price Data (comma-separated, e.g., 10, 15, 20)", "")
        opening_data = st.text_area("Opening Price Data (comma-separated, e.g., 10, 15, 20)", "")
        high_data = st.text_area("High Price Data (comma-separated, e.g., 10, 15, 20)", "")
        low_data = st.text_area("Low Price Data (comma-separated, e.g., 10, 15, 20)", "")
        volume_data = st.text_area("Volume Data (comma-separated, e.g., 1000, 1500, 2000)", "")

        # Check if all fields are filled
        if closing_data and opening_data and high_data and low_data and volume_data:
            try:
                # Convert the text data to lists of floats
                closing_price_list = [float(x.strip()) for x in closing_data.split(",")]
                opening_price_list = [float(x.strip()) for x in opening_data.split(",")]
                high_price_list = [float(x.strip()) for x in high_data.split(",")]
                low_price_list = [float(x.strip()) for x in low_data.split(",")]
                volume_list = [float(x.strip()) for x in volume_data.split(",")]

                # Combine the lists into a DataFrame
                stock_price_data = pd.DataFrame({
                    "Closing Price": closing_price_list,
                    "Opening Price": opening_price_list,
                    "High Price": high_price_list,
                    "Low Price": low_price_list,
                    "Volume": volume_list,
                })

                st.write("Data Preview:", stock_price_data.head())

            except ValueError:
                st.error("Please ensure all data is properly formatted as comma-separated numerical values.")
        else:
            st.warning("Please fill out all fields to proceed.")


    if 'data' in locals() and 'stock_price_columns' in locals():
        if st.button("Forecast Stock Prices"):





            forecast, prompt = forecast_stock_price(data, stock_price_columns)
            if forecast is not None:    
                st.write("Forecasted Stock Prices:", forecast)

                explanation = generate_explanation(data, forecast)
                st.write("Explanation:", explanation)

                # Visualization
                plt.figure(figsize=(10, 5))
                for column in ['Closing Price', 'Opening Price', 'High Price', 'Low Price']:
                    plt.plot(data.index, data[column], label=f'{column}', marker='o')

                # Since volume might have a different scale, we plot it on a secondary y-axis
                ax = plt.gca()  # Get current axis
                ax2 = ax.twinx()  # Create another axis that shares the same x-axis
                ax2.plot(data.index, data['Volume'], label='Volume', color='grey', linestyle='--')
                ax2.set_ylabel('Volume')

                # Labeling the plot
                plt.title('Stock Price Data')
                plt.xlabel('Date')
                plt.ylabel('Price')

                # Adding a legend to show labels
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, loc='upper left')

                # Display the plot
                plt.show()
