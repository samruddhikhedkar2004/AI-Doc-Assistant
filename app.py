import streamlit as st
import boto3
import json
import os
from pypdf import PdfReader

# Page Title
st.title("📄 AI Document Assistant")
st.write("Upload a PDF and ask questions about the document using Generative AI.")

# Sidebar
st.sidebar.title("About")
st.sidebar.write(
    "This demo shows how Generative AI can answer questions from documents using Amazon Bedrock."
)
st.sidebar.write("Model: Amazon Nova Lite")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    st.success("Document uploaded successfully")

    reader = PdfReader(uploaded_file)

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # limit text length for Bedrock
    text = text[:15000]

    # Preview document
    with st.expander("Preview Document Text"):
        st.write(text[:2000] + "...")

    # Summarize button
    if st.button("Summarize Document"):
        question = "Provide a concise summary of this document."
    else:
        question = None

    # Question form
    with st.form("question_form"):
        user_question = st.text_input("Ask a question about the document")
        submit_button = st.form_submit_button("Ask AI")

        if submit_button:
            question = user_question

    if question:

        prompt = f"""
Answer the question using the document below.

Document:
{text}

Question:
{question}
"""

        # ✅ UPDATED: Use environment variables (for deployment)
        client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION")
        )

        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        # Spinner while processing
        with st.spinner("AI is analyzing the document..."):

            response = client.invoke_model(
                modelId="amazon.nova-lite-v1:0",
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )

            result = json.loads(response["body"].read())
            answer = result["output"]["message"]["content"][0]["text"]

        # Clean response display
        st.markdown("### AI Response")
        st.markdown(answer)