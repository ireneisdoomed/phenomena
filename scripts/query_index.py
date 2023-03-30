# load environment variables from .env file
from dotenv import load_dotenv
import os
import pinecone

load_dotenv()

def load_index():
    pinecone_api_key = os.getenv("PINECONE_TOKEN")
    pinecone.init(api_key=pinecone_api_key, environment="us-west4-gcp")
    return pinecone.Index("phenomena")

def query_index_by_id(id: str):
    """Query the index by id. The endpoint accepts lists to be provided."""
    return index.fetch([id])

if __name__ == "__main__":
    index = load_index()
    usr_input = input("Enter an EFO id: ")
    res = query_index_by_id(usr_input)
    vector = res["vectors"][usr_input]
    print(vector)
    # TOFIX: not sure about the response format

