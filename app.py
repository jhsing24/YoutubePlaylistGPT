from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone
import config
from sys import exit
from url_grabber import url_grabber

llm = OpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY)
embeddings=OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
pinecone.init(
    api_key=config.PINECONE_API_KEY,
    environment=config.PINECONE_API_ENV
)

index_name="playlistgpt"
print("Welcome to PlaylistGPT!")
print('\n\n')
valid_url = False

while(not valid_url):
    playlist_url = input("Input the playlist you would like to analyze (or type 'quit' to exit): ")
    if "youtube.com" in playlist_url and "playlist" in playlist_url:
        valid_url = True
    elif playlist_url == "quit":
        print("Quitting...")
        exit(0)
    else:
        print("There was an error with the provided URL.")

videoList = url_grabber(playlist_url)
print(f"Your playlist has {len(videoList)} videos!")
print("Loading...")
texts = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
for url in videoList:
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    res = loader.load()
    texts.extend(text_splitter.split_documents(res))

db = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":3})
print("\n\nDone analyzing!")
print("Ready to answer some questions! (Type 'quit' to exit any time)")
run_queries = True
while(run_queries):
    print()
    query = input("Ask me a question: ")
    if query == "quit":
        print("Quitting...")
        exit(0)
    else:
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        result = qa({"query": query})
        print(result["result"])
        # TODO: Sources cannot be determined, as pytube cannot read titles for some reason
        # print("Source(s): ")
        # for source in result['source_documents']:
        #     print(source.metadata.get('source'))