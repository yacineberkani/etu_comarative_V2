from __init__ import *



nltk.download('stopwords')

# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Fonction pour prétraiter le texte pour l'analyse des topics
def preprocess_text_for_topics(text):
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'\b\d{4}\b', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()

    stop_words = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)
    stop_words = list(stop_words)

    def is_valid_token(token):
        return len(token) > 2 and token not in stop_words

    tokens = [word for word in text.split() if is_valid_token(word)]
    cleaned_text = ' '.join(tokens)

    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=500, ngram_range=(1, 2))
    vectorizer.fit_transform([cleaned_text])

    return cleaned_text, vectorizer

# Fonction pour extraire et analyser les topics
def extract_and_analyze(pdf_path):
    extracted_text = extract_text_from_pdf(pdf_path)
    preprocessed_text, vectorizer = preprocess_text_for_topics(extracted_text)
    
    num_topics = 5
    nmf = NMF(n_components=num_topics, random_state=42).fit(vectorizer.transform([preprocessed_text]))
    
    topics = []
    topic_weights = []
    for topic_idx, topic in enumerate(nmf.components_):
        topic_words = [vectorizer.get_feature_names_out()[i] for i in sorted(range(len(topic)), key=lambda i: topic[i])[-5:]]
        topics.extend(topic_words)
        weights = sorted(topic)[-5:]
        topic_weights.extend(weights)

    topics = [x for _, x in sorted(zip(topic_weights, topics), reverse=True)]
    topic_weights = sorted(topic_weights, reverse=True)
    
    return topics[:7], topic_weights[:7]

def cosine_similarity_keywords(topics_list):
    """
    Calcule la matrice de similarité cosinus entre plusieurs listes de mots-clés.
    
    Args:
    topics_list (list of list of str): Listes de mots-clés pour chaque document.
    
    Returns:
    np.array: Matrice de similarité cosinus.
    """
    vectorizer = CountVectorizer().fit_transform([' '.join(topics) for topics in topics_list])
    vectors = vectorizer.toarray()
    cosine_sim_matrix = cosine_similarity(vectors)
    
    return cosine_sim_matrix

def compare_top_topics(pdf_paths):
    all_topics = []
    all_weights = []
    
    for pdf_path in pdf_paths:
        topics, weights = extract_and_analyze(pdf_path)
        all_topics.append(topics)
        all_weights.append(weights)
    
    similarity_matrix = cosine_similarity_keywords(all_topics)
    
    num_pdfs = len(pdf_paths)
    num_cols = 5
    num_rows = (num_pdfs + num_cols - 1) // num_cols  # Calculer le nombre de lignes nécessaires

    fig_topics, axs_topics = plt.subplots(num_rows, num_cols, figsize=(20, 8 * num_rows))

    axs_topics = axs_topics.flatten()

    for i, (topics, weights, ax) in enumerate(zip(all_topics, all_weights, axs_topics)):
        sns.barplot(x=weights, y=topics, ax=ax, palette="viridis")
        ax.set_title(f'Topics for PDF {i+1}')
        ax.set_xlabel('Poids')
        ax.set_ylabel('Topics')

    # Cacher les axes vides si le nombre de PDFs est inférieur à num_rows * num_cols
    for j in range(i + 1, len(axs_topics)):
        fig_topics.delaxes(axs_topics[j])

    fig_topics.tight_layout()
    fig_topics.savefig('topics.png')
    plt.close(fig_topics)

    fig_sim, ax_sim = plt.subplots(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap="viridis", xticklabels=[f'Doc {i+1}' for i in range(num_pdfs)], yticklabels=[f'Doc {i+1}' for i in range(num_pdfs)], ax=ax_sim)
    ax_sim.set_title('Cosine Similarity Matrix')

    fig_sim.tight_layout()
    fig_sim.savefig('similarity_matrix.png')
    plt.close(fig_sim)

    return all_topics, similarity_matrix



def save_results_to_file(all_topics, similarity_matrix, file_path):
    with open(file_path, 'w') as file:
        for i, topics in enumerate(all_topics):
            file.write(f"Topics for PDF {i+1}:\n")
            file.write(", ".join(topics) + "\n\n")
        
        file.write("Similarity Matrix:\n")
        file.write(np.array2string(similarity_matrix, precision=2, separator=', ') + "\n")
