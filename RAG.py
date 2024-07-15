from __init__ import *


# Définition des constantes
HF_TOKEN = "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"  # Remplacer par votre API HuggingFace
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct" 
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
QUERY = "Please do a comparative study on these PDF documents"



# Téléchargement des stopwords et des tokenizers
nltk.download("stopwords")
nltk.download("punkt")

# Fonction pour extraire du texte d'un PDF en utilisant slate
def extract_text(file):
    pdfFileObj = open(file, "rb")
    pdfPages = slate.PDF(pdfFileObj)
    text = ""
    for page in pdfPages:
        text += page
    return text

# Fonction pour extraire du texte d'un PDF en utilisant OCR
def extract_ocr(file):
    pages = convert_from_path(file, 500)

    image_counter = 1
    for page in pages:
        filename = "page_" + str(image_counter) + ".jpg"
        page.save(filename, "JPEG")
        image_counter = image_counter + 1

    limit = image_counter - 1
    text = ""
    for i in range(1, limit + 1):
        filename = "page_" + str(i) + ".jpg"
        page = str(((pytesseract.image_to_string(Image.open(filename)))))
        page = page.replace("-\n", "")
        text += page
        os.remove(filename)
    return text

# Fonction pour résumer le texte extrait
def summarize(text):
    processedText = re.sub("’", "'", text)
    processedText = re.sub("[^a-zA-Z' ]+", " ", processedText)
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(processedText)

    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        elif stemmer.stem(word) in freqTable:
            freqTable[stemmer.stem(word)] += 1
        else:
            freqTable[stemmer.stem(word)] = 1

    sentences = sent_tokenize(text)
    stemmedSentences = []
    sentenceValue = dict()
    for sentence in sentences:
        stemmedSentence = []
        for word in sentence.lower().split():
            stemmedSentence.append(stemmer.stem(word))
        stemmedSentences.append(stemmedSentence)

    for num in range(len(stemmedSentences)):
        for wordValue in freqTable:
            if wordValue in stemmedSentences[num]:
                if sentences[num][:12] in sentenceValue:
                    sentenceValue[sentences[num][:12]] += freqTable.get(wordValue)
                else:
                    sentenceValue[sentences[num][:12]] = freqTable.get(wordValue)

    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue.get(sentence)

    average = int(sumValues / len(sentenceValue))

    summary = ""
    for sentence in sentences:
        if sentence[:12] in sentenceValue and sentenceValue[sentence[:12]] > (3.0 * average):
            summary += " " + " ".join(sentence.split())

    summary = re.sub("’", "'", summary)
    summary = re.sub("[^a-zA-Z0-9'\"():;,.!?— ]+", " ", summary)

    return summary

# Remplacement de la méthode de résumé existante par la vôtre
def extract_and_summarize_pdf(file_path, method="text"):
    if method == "text":
        text = extract_text(file_path)
    elif method == "ocr":
        text = extract_ocr(file_path)
    else:
        raise ValueError("Invalid method specified. Use 'text' or 'ocr'.")
    
    summary = summarize(text)
    return summary

# Fonction pour lire les PDFs et extraire les résumés
def read_pdfs(pdf_paths, method="text"):
    documents = []
    for pdf_path in pdf_paths:
        summary = extract_and_summarize_pdf(pdf_path, method)
        document = Document(text=summary, metadata={"source": pdf_path})
        documents.append(document)
    return documents



def setup_llm():
    return HuggingFaceInferenceAPI(model_name=MODEL_NAME, token=HF_TOKEN)

def setup_embed_model():
    return LangchainEmbedding(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))


def setup_service_context(llm):
    Settings.llm = llm
    Settings.chunk_size = 1200

def setup_storage_context():
    graph_store = SimpleGraphStore()
    return StorageContext.from_defaults(graph_store=graph_store)

def construct_knowledge_graph_index(documents, storage_context, embed_model):
    return KnowledgeGraphIndex.from_documents(
        documents=documents,
        max_triplets_per_chunk=5,
        storage_context=storage_context,
        embed_model=embed_model,
        include_embeddings=True
    )

def create_query_engine(index):
    return index.as_query_engine(
        include_text=True,
        response_mode="tree_summarize",
        embedding_mode="hybrid",
        similarity_top_k=3,
    )

def generate_response(query_engine, query):
    response = query_engine.query(query)
    return response

def save_response(response, filename="response.md"):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(response.response.split("<|assistant|>")[-1].strip())
    print("Le document a été sauvegardé avec succès")

def visualize_knowledge_graph(index, output_html="Knowledge_graph.html"):
    g = index.get_networkx_graph()
    net = Network(notebook=True, cdn_resources="in_line", directed=True)
    net.from_nx(g)
    net.save_graph(output_html)
    display(HTML(filename=output_html))
    print("Le graphe de connaissances a été sauvegardé et affiché avec succès.")

