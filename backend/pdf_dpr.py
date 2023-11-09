from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PreProcessor, PDFToTextConverter
from haystack.nodes.retriever.dense import DensePassageRetriever
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import FARMReader
from haystack.utils import convert_files_to_docs
from tqdm.auto import tqdm
from haystack.utils import print_answers


# Initialize the document store
document_store = InMemoryDocumentStore()

# Convert files to dicts containing the text of the documents
# dicts = convert_files_to_docs(dir_path='F:\Projects\dpr-app\backend\deephashing.pdf')
# print(len(dicts))

pdf_converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["en"])
doc_pdf = pdf_converter.convert(file_path='deephashing.pdf', meta=None)
# print(doc_pdf)  # This should show the content of the PDF if conversion is successful.

# Preprocess documents (e.g., split into smaller chunks)
# We recommend you split the text from your files into small documents of around 100 words for dense retrieval methods
preprocessor = PreProcessor(split_length=100, split_overlap=0, split_respect_sentence_boundary=True)
processed_docs = []
for doc in tqdm(doc_pdf, desc="Processing documents"):
    processed = preprocessor.process([doc])  # process each document individually
    processed_docs.extend(processed)  # extend the list with the results

document_store.write_documents(processed_docs)

# Initialize the DPR Retriever with GPU enabled
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=True
)

# Update the embeddings for our documents in the document store
document_store.update_embeddings(retriever)

# Initialize a reader with GPU enabled
model_name_or_path = "deepset/roberta-base-squad2"
reader = FARMReader(model_name_or_path, use_gpu=True)

# Initialize the Extractive QA Pipeline
pipe = ExtractiveQAPipeline(reader=reader, retriever=retriever)

# Make a query
prediction = pipe.run(query="What is LSH?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})

# Output the results
if prediction['answers']:
    for idx, answer_obj in enumerate(prediction['answers']):
        answer = answer_obj.__dict__  # Convert the Answer object to a dictionary if necessary
        print(f"\nAnswer {idx+1}:")
        print(f"  Text: {answer['answer']}")
        print(f"  Score: {answer['score']:.4f}")
        context = answer['context'] if 'context' in answer else "Not provided"
        print(f"  Context: {context}")
        doc_id = answer['document_ids'][0] if 'document_ids' in answer else "Not provided"
        print(f"  Document ID: {doc_id}")
        # Handle Span object or dictionary for offsets
        start_pos = answer['offsets_in_document'][0].start if hasattr(answer['offsets_in_document'][0], 'start') else answer['offsets_in_document'][0]['start']
        end_pos = answer['offsets_in_document'][0].end if hasattr(answer['offsets_in_document'][0], 'end') else answer['offsets_in_document'][0]['end']
        print(f"  Start: {start_pos}, End: {end_pos}")
else:
    print("\nNo answers found.")

print_answers(results, details="all")

