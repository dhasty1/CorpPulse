# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import streamlit as st
from pipeline import rag_pipeline
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

pipeline = rag_pipeline()

# Function to execute the RAG pipeline
def generate_answer(query: str):
    result = pipeline.run(
        {
            "text_embedder": {"text": query},
            "prompt_builder": {"question": query}
        }
    )
    answer = result["llm"]["replies"][0]
    return str(answer).replace("$", "\$")

def stream_answer(answer):
    for word in answer.split(" "):
        yield word + " "
        time.sleep(0.02)


#################################
#    Begin Streamlit UI Code    #
#################################

def run():
    # Sidebar for uploading new documents
    st.sidebar.title("Upload a New File")
    uploaded_file = st.sidebar.file_uploader("Choose a .txt file", type="txt")

    if uploaded_file is not None:
      # Get the file name
      filename = uploaded_file.name

      # Save the uploaded file to the data directory
      with open(os.path.join("data/Letters", filename), "wb") as f:
          f.write(uploaded_file.getbuffer())

      st.sidebar.success('File uploaded successfully!')

    # Set column layout for header
    col1, col2 = st.columns(2)
    with col1:
        col1.image("img/header-small.svg", use_column_width=True)
    with col2:
        col2.markdown("<h3 style='text-align: right; margin-top: 20%;'>Query a dataset of over 125 shareholder letters</h3>", unsafe_allow_html=True)

    st.divider()

    user_question = st.chat_input("Ask us a question:")

    if user_question:
        user = st.chat_message("human")
        user.write(user_question)
        message = st.chat_message("assistant")
        with st.spinner("Thinking..."):
            answer = generate_answer(user_question)
        message.write("Hello, ")
        message.write_stream(stream_answer(answer))

if __name__ == "__main__":
    run()
