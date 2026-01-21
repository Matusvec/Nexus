### Language model augmentation
You can use language models to create chunks. For example, use a large language model, such as GPT-4, to generate textual representations of images or summaries of tables that become chunks. Language model augmentation is often used with other chunking approaches such as custom code.

If your document analysis determines that the text before or after the image helps answer some requirement questions, pass this extra context to the language model. It's important to experiment to determine whether this extra context improves the performance of your solution.

If your chunking logic splits the image description into multiple chunks, include the image URL in each chunk to ensure that metadata is returned for all queries that the image serves. This step is crucial for scenarios where the user needs to access the source image through that URL or use raw images during inferencing time.

Tools: Azure OpenAI, OpenAI
Engineering effort: Medium
Processing cost: High
Use cases: Images, tables
Examples: Generate text representations of tables and images, summarize transcripts from meetings, speeches, interviews, or podcasts


### Fixed-size parsing, with overlap
This approach breaks down a document into chunks based on a fixed number of characters or tokens and allows overlap of characters between chunks. This approach has many of the same advantages and disadvantages as sentence-based parsing. One advantage of this approach over sentence-based parsing is the ability to obtain chunks with semantic meanings that span multiple sentences.

You must choose the fixed size of the chunks and the amount of overlap. Because the results vary for different document types, it's best to use a tool like the Hugging Face chunk visualizer to do exploratory analysis. You can use tools like this to visualize how your documents are chunked based on your decisions. You should use bidirectional encoder representations from transformers (BERT) tokens instead of character counts when you use fixed-sized parsing. BERT tokens are based on meaningful units of language, so they preserve more semantic information than character counts.

Tools: LangChain recursive text splitter, Hugging Face chunk visualizer
Engineering effort: Low
Processing cost: Low
Use cases: Unstructured documents written in prose or nonprose with complete or incomplete sentences. Your collection of documents contains a prohibitive number of different document types that require individual chunking strategies.
Examples: User-generated content like open-ended feedback from surveys, forum posts, reviews, email messages, personal notes, research notes, and lists

### Semantic chunking
This approach uses embeddings to group conceptually similar content across a document to create chunks. Semantic chunking can produce easily understandable chunks that closely align to the content's subjects. The logic for this approach can search a document or set of documents to find recurring information and create chunks that group the mentions or sections together. This approach can be more costly because it requires you to develop complex custom logic.

Tools: Custom implementation. Natural language processing (NLP) tools like spaCy can help with sentence-based parsing.
Engineering effort: High
Processing cost: High
Use cases: Documents that have topical overlap throughout their sections
Examples: Financial or healthcare-focused documentation


### Implementation for Tables in Documents during document processing:
The Solution:
I approached this with four key concepts:

Precise Extraction: Cleanly extract all tables from the document.
Contextual Enrichment: Leverage an LLM to generate a robust, contextual description of each table by analyzing both the extracted table and its surrounding document content.
Format Standardization: Employ an LLM to convert tables into a uniform markdown format, enhancing both embedding efficiency and LLM comprehension.
Unified Embedding: Create a ‘table chunk’ by combining the contextual description with the markdown-formatted table, optimizing it for vector database storage and retrieval.

Each table chunk will have the contextualized description of the the table and the table in makrdown format.


Step 1: Precise Extraction
To begin, we need to extract text and tables from the document, to do this we will use Unstructured.io.


# ALL of this below is just a suggestion on how to do things, i dont want to follow everything here, i have many diffferences in things i am using, but look at this and take advantabge of what this man figured out with tables in RAG, we have many differences, like im using gemini, other database, ect, take all you can from this

Let’s install and import all dependencies:

!apt-get -qq install poppler-utils tesseract-ocr
%pip install -q --user --upgrade pillow
%pip install -q --upgrade unstructured["all-docs"]
%pip install kdbai_client
%pip install langchain-openai
%pip install langchain
import os
!git clone -b KDBAI_v1.4 https://github.com/KxSystems/langchain.git
os.chdir('langchain/libs/community')
!pip install .
%pip install pymupdf
%pip install --upgrade nltk

import os
from getpass import getpass
import openai
from openai import OpenAI
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.auto import partition
from langchain_openai import OpenAIEmbeddings
import kdbai_client as kdbai
from langchain_community.vectorstores import KDBAI
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import fitz
nltk.download('punkt')
Set OpenAI API key:

# Set OpenAI API
if "OPENAI_API_KEY" in os.environ:
    KDBAI_API_KEY = os.environ["OPENAI_API_KEY"]
else:
    # Prompt the user to enter the API key
    OPENAI_API_KEY = getpass("OPENAI API KEY: ")
    # Save the API key as an environment variable for the current session
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
Download Meta’s second quarter 2024 results PDF (lots of tables!):

!wget 'https://s21.q4cdn.com/399680738/files/doc_news/Meta-Reports-Second-Quarter-2024-Results-2024.pdf' -O './doc1.pdf'
We will use Unstructured’s ‘partition_pdf’ implementing the ‘hi_res’ partitioning strategy to extract text and table elements from the PDF earnings report.

There are a few parameters we can set during partitioning to ensure we extract tables accurately from the PDF.

strategy = “hi_res”: Identifies the layout of the document, recommended for use-cases sensitive to correct element classification, for example table elements.
chunking_strategy = “by_title”: The ‘by_title’ chunking strategy preserves section boundaries by starting a new chunk when a ‘Title’ element is encountered, even if the current chunk has space, ensuring that text from different sections doesn’t appear in the same chunk.
elements = partition_pdf('./doc1.pdf',
                              strategy="hi_res",
                              chunking_strategy="by_title",
                              )
Let’s see what elements have been extracted:

from collections import Counter
display(Counter(type(element) for element in elements))
>>> Counter({unstructured.documents.elements.CompositeElement: 17,
         unstructured.documents.elements.Table: 10})
There are 17 CompositeElement elements extracted, which are basically text chunks. There are 10 Table elements, which are the extracted tables.

At this point, we have extracted text chunks and tables from the document.

Step 2 & 3: Table Contextual Enrichment and Format Standardization
Let’s take a look at a Table element to see if we can understand why there might be issues with this in a RAG pipeline. The second to last element is a Table element:

print(elements[-2])
>>>Foreign exchange effect on 2024 revenue using 2023 rates Revenue excluding foreign exchange effect GAAP revenue year-over-year change % Revenue excluding foreign exchange effect year-over-year change % GAAP advertising revenue Foreign exchange effect on 2024 advertising revenue using 2023 rates Advertising revenue excluding foreign exchange effect 2024 $ 39,071 371 $ 39,442 22 % 23 % $ 38,329 367 $ 38,696 22 % 2023 $ 31,999 $ 31,498 2024 $ 75,527 265 $ 75,792 25 % 25 % $ 73,965 261 $ 74,226 24 % 2023 GAAP advertising revenue year-over-year change % Advertising revenue excluding foreign exchange effect year-over-year change % 23 % 25 % Net cash provided by operating activities Purchases of property and equipment, net Principal payments on finance leases $ 19,370 (8,173) (299) $ 10,898 $ 17,309 (6,134) (220) $ 10,955 $ 38,616 (14,573) (614) $ 23,429
We see that the table is represented as a long string with a mix of natural language and numbers. If we just used this as our table chunk to be ingested to the RAG pipeline, it is easy to see how it would be difficult to decipher if this table should be retrieved or not.

We need to enrich each table with context, and then format the table into markdown.

To do this we will first extract the entire text from the pdf document to be used as context:

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

pdf_path = './doc1.pdf'
document_content = extract_text_from_pdf(pdf_path)
Next, create a function that will take in the entire context of the document (from the above code), along with the extracted text of a specific table, and output a new description containing a comprehensive description of the table, and the table itself transformed into markdown format:

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_table_description(table_content, document_context):
    prompt = f"""
    Given the following table and its context from the original document,
    provide a detailed description of the table. Then, include the table in markdown format.

    Original Document Context:
    {document_context}

    Table Content:
    {table_content}

    Please provide:
    1. A comprehensive description of the table.
    2. The table in markdown format.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that describes tables and formats them in markdown."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
Now put it all together by applying the above function to all Table elements, and replacing each original Table element’s text with the new description (consisting of the contextual description of the table and markdown formatted table):

# Process each table in the directory
for element in elements:
  if element.to_dict()['type'] == 'Table':
    table_content = element.to_dict()['text']

    # Get description and markdown table from GPT-4o
    result = get_table_description(table_content, document_content)
    # Replace each Table elements text with the new description
    element.text = result

print("Processing complete.")
Example of a enriched Table chunk/element (in markdown format for easy reading):

This markdown table provides a concise presentation of the financial data, making it easy to read and comprehend in a digital format.
### Detailed Description of the Table

The table presents segment information from Meta Platforms, Inc. for both revenue and income (loss) from operations. The data is organized into two main sections: 
1. **Revenue**: This section is subdivided into two categories: "Advertising" and "Other revenue". The total revenue generated from these subcategories is then summed up for two segments: "Family of Apps" and "Reality Labs". The table provides the revenue figures for three months and six months ended June 30, for the years 2024 and 2023.
2. **Income (loss) from operations**: This section shows the income or loss from operations for the "Family of Apps" and "Reality Labs" segments, again for the same time periods.

The table allows for a comparison between the two segments of Meta's business over time, illustrating the performance of each segment in terms of revenue and operational income or loss. 

### The Table in Markdown Format

```markdown
### Segment Information (In millions, Unaudited)

|                            | Three Months Ended June 30, 2024 | Three Months Ended June 30, 2023 | Six Months Ended June 30, 2024 | Six Months Ended June 30, 2023 |
|----------------------------|----------------------------------|----------------------------------|------------------------------- |-------------------------------|
| **Revenue:**               |                                  |                                  |                               |                               |
| Advertising                | $38,329                          | $31,498                          | $73,965                       | $59,599                       |
| Other revenue              | $389                             | $225                             | $769                          | $430                          |
| **Family of Apps**         | $38,718                          | $31,723                          | $74,734                       | $60,029                       |
| Reality Labs               | $353                             | $276                             | $793                          | $616                          |
| **Total revenue**          | $39,071                          | $31,999                          | $75,527                       | $60,645                       |
|                            |                                  |                                  |                               |                               |
| **Income (loss) from operations:** |                                  |                                  |                               |                               |
| Family of Apps             | $19,335                          | $13,131                          | $36,999                       | $24,351                       |
| Reality Labs               | $(4,488)                         | $(3,739)                         | $(8,334)                      | $(7,732)                      |
| **Total income from operations** | $14,847                          | $9,392                           | $28,665                       | $16,619                       |
```
As you can see, this provides much more context than the Table element’s original text which should significantly improve the performance of our RAG pipeline. We now have fully contextualized Table chunks that can be prepared for retrieval by embedding and storing them in our vector database.

Step 4: Unified Embeddings… Prepare for RAG
Now that all of the elements have the necessary context for high quality retrieval and generation, we will take our elements, embed them, and store them in the KDB.AI vector database.

First, we will create embeddings for each element, embeddings are just a numerical representation of the semantic meaning of each element:

from unstructured.embed.openai import OpenAIEmbeddingConfig, OpenAIEmbeddingEncoder

embedding_encoder = OpenAIEmbeddingEncoder(
    config=OpenAIEmbeddingConfig(
      api_key=os.getenv("OPENAI_API_KEY"),
      model_name="text-embedding-3-small",
    )
)
elements = embedding_encoder.embed_documents(
    elements=elements
)
Next, create a Pandas DataFrame to store our elements within. The DataFrame will contain columns based on the attributes of each element extracted with Unstructured. For example, Unstructured creates an ID, text (which we manipulated for the Table elements), metadata, and embedding (created above) for each element. We store this data in a DataFrame as this format is easily ingestible into the KDB.AI vector database.

import pandas as pd
data = []

for c in elements:
  row = {}
  row['id'] = c.id
  row['text'] = c.text
  row['metadata'] = c.metadata.to_dict()
  row['embedding'] = c.embeddings
  data.append(row)

df = pd.DataFrame(data)
Setup KDB.AI Server:

Get Ryan Siegler’s stories in your inbox
Join Medium for free to get updates from this writer.

Enter your email
Subscribe
Sign-up for KDB.AI server for free here: https://trykdb.kx.com/kdbai/signup/

## start session with KDB.AI Server
session = kdbai.Session(endpoint="http://localhost:8082")
You are now connected to the vector database instance — the next step is to define the schema for the table you will create within KDB.AI:

schema = [
    {'name': 'id', 'type': 'str'},
    {'name': 'text', 'type': 'bytes'},
    {'name': 'metadata', 'type': 'general'},
    {'name': 'embedding', 'type': 'float32s'}
]
We create a column in the schema for each column in the DataFrame created earlier. (id, text, metadata embedding). The embedding column is where the vector search for retrieval will be executed.

Next, we define the index. Several parameters are defined here:

name: the user defined name of this index.
column: the column within the above schema that this index will be applied to. In this case, the ‘embedding’ column.
type: the type of index, here simply using a flat index, but could also use qFlat(on-disk flat index), HNSW, IVF, IVFPQ.
params: the dims and vector search metric used. dims is the number of dimensions of each embedding — determined by which embedding model is used. In this case OpenAI’s ‘text-embedding-3-small’ outputs 1536 dimension embeddings. The metric, L2, is Euclidean distance, other options include cosine similarity and dot product.
indexes = [
       {'name': 'flat_index', 
        'column': 'embedding', 
        'type': 'flat', 
        'params': {'dims': 1536, 'metric': 'L2'}}
]
Table creation based on the above schema:

# Connect to the default database in KDB.AI
database = session.database('default')

KDBAI_TABLE_NAME = "Table_RAG"

# First ensure the table does not already exist
if KDBAI_TABLE_NAME in database.tables:
    database.table(KDBAI_TABLE_NAME).drop()

#Create the table using the table name, schema, and indexes defined above
table = db.create_table(table=KDBAI_TABLE_NAME, schema=schema, indexes=indexes)
Insert the DataFrame into the KDB.AI table:

# Insert Elements into the KDB.AI Table
table.insert(df)
All elements are now stored in the vector database which is ready to be queried for retrieval.

Use LangChain and KDB.AI to Perform RAG!
Basic setup for using LangChain:

# Define OpenAI embedding model for LangChain to embed the query
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# use KDBAI as vector store
vecdb_kdbai = KDBAI(table, embeddings)
Define a RAG chain using KDB.AI as the retriever and gpt-4o as the LLM for generation:

# Define a Question/Answer LangChain chain
qabot = RetrievalQA.from_chain_type(
    chain_type="stuff",
    llm=ChatOpenAI(model="gpt-4o"),
    retriever=vecdb_kdbai.as_retriever(search_kwargs=dict(k=5, index='flat_index')),
    return_source_documents=True,
)
Helper function to perform RAG:

# Helper function to perform RAG
def RAG(query):
  print(query)
  print("-----")
  return qabot.invoke(dict(query=query))["result"]