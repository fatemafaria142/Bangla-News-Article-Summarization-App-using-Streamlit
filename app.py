# Import necessary libraries
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from normalizer import normalize

# Set the page configuration
st.set_page_config(
    page_title="Bangla Abstractive Summarization App",  # Title of the app displayed in the browser tab
    page_icon=":shield:",  # Path to a favicon or emoji to be displayed in the browser tab
    initial_sidebar_state="auto"  # Initial state of the sidebar ("auto", "expanded", or "collapsed")
)

# Load custom CSS styling
with open("assets/style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

# Function to load the pre-trained model
@st.cache_resource(experimental_allow_widgets=True)
def get_model(): #pip install git+https://github.com/csebuetnlp/normalizer
    tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglat5", use_fast=True) 
    model = AutoModelForSeq2SeqLM.from_pretrained("Soyeda10/BanglaTextSummarization") 
    return tokenizer, model

# Load the tokenizer and model
tokenizer, model = get_model()


# Add a header to the Streamlit app
st.header("BanglaSynopsi")

# Create a columns layout with width ratios 2:1
col1, col2 = st.columns([2, 2])

# Add user input text area in the first column (col1)
with col1:
    st.markdown("<span style='color:black'>Original Text</span>", unsafe_allow_html=True)
    user_input = st.text_area("Enter your text here", "", height=450, label_visibility="collapsed", key="user_input")



# Add a text area for summarized text in the second column (col2)
with col2:
    st.markdown("<span style='color:black'>Summarized Text</span>", unsafe_allow_html=True)
    summarized_text = st.empty()  # Create an empty container for summarized text

# Add the button inside the row
submit_button = st.button("Summarize")


# Perform summarization when user input is provided and the submit button is clicked
if user_input and submit_button:
    input_ids = tokenizer(normalize(user_input), padding=True, truncation=True, max_length=512, return_tensors="pt").input_ids
    generated_tokens = model.generate(input_ids, max_new_tokens=512)  # Set max_new_tokens to control generation length
    decoded_tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    summarized_text.write(decoded_tokens)  # Write the generated summary to the summarized_text widget