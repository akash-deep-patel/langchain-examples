import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Streamlit app
st.subheader('Summarize URL')

# Get OpenAI API key and URL to be summarized
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API key", value="", type="password")
url = st.text_input("URL", label_visibility="collapsed")

# If 'Summarize' button is clicked
if st.button("Summarize"):
    # Validate inputs
    if not openai_api_key.strip() or not url.strip():
        st.error("Please provide the missing fields.")
    elif not validators.url(url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Please wait..."):
                # Load URL data
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[url], ssl_verify=False, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                data = loader.load()
                print(type(data))
                print(type(data[0]))
                # print(article)
                # Check if data is a tuple 
                article_content = ""
                # Extract content based on type 
                if isinstance(data[0], tuple): 
                    article_content = data[0][0] # Extract the first element of the tuple 
                    print("tuple") 
                elif isinstance(data[0], str): 
                    print("str") 
                    article_content = data[0] 
                elif hasattr(data[0], 'page_content'): 
                    print("document") 
                    article_content = data[0].page_content 
                    print(article_content[:10]) 
                else: 
                    raise ValueError("Unsupported data format")

                # Ensure article_content is a string 
                if not isinstance(article_content, str): 
                    raise ValueError("Extracted content is not a string")
                
                # Initialize the ChatOpenAI module, load and run the summarize chain
                # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key)
                llm = ChatOllama(temperature=0, model="llama3.2")
                prompt_template = """Write a summary of the following in 250-300 words:
                    
                    {text}

                """
                prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                
                print(prompt)
                # summary = chain.run(text=article_content)
                summary = chain.run({"input_documents": [data[0]]})
                print(summary)

                st.success(summary)
        except Exception as e:
            st.exception(f"Exception: {e}")
