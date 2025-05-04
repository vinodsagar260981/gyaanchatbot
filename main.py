from langchain.llms.bedrock import Bedrock
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import boto3
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv('.env', override=True)

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY= os.getenv("AWS_SECRET_ACCESS_KEY")
region_name=os.getenv("AWS_DEFAULT_REGION")

#bedrock client 
bedrock = boto3.client(
    service_name = "bedrock-runtime",
    region_name = region_name,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

model_id = "mistral.mistral-7b-instruct-v0:2"

llm = Bedrock(
    model_id=model_id,
    client=bedrock,
    model_kwargs={"temperature": 0.9}
)

#prompt templates
def my_chatbot(language, user_text):
    prompt = PromptTemplate(
        input_variables=["language", "user_text"],
        template="You are a chatbot. You are in {language}.\n\n{user_text}"
    )
    
    bedrock_chain = LLMChain(llm=llm, prompt=prompt)
    response = bedrock_chain({'language':language, 'user_text':user_text})
    
    return response


st.title("Gyaan Chat")

language = st.sidebar.selectbox("language", ["english", "spanish", "hindi", "kannada"])

if language:
    user_text = st.sidebar.text_area(label="what is your question?", max_chars=100)
    
if user_text:
    response = my_chatbot(language, user_text)
    st.write(response['text'])
    

