from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Fonctions d'évaluation

def evaluate_accuracy(reference_responses, generated_responses):
    """
    Evaluate the accuracy of generated responses compared to reference responses.
    """
    correct = 0
    for ref, gen in zip(reference_responses, generated_responses):
        if ref == gen:
            correct += 1
    accuracy = correct / len(reference_responses)
    return accuracy

def evaluate_bleu(reference_responses, generated_responses):
    """
    Evaluate the BLEU score of generated responses compared to reference responses.
    """
    scores = []
    for ref, gen in zip(reference_responses, generated_responses):
        score = sentence_bleu([ref.split()], gen.split())
        scores.append(score)
    return np.mean(scores)

def evaluate_rouge(reference_responses, generated_responses):
    """
    Evaluate the ROUGE score of generated responses compared to reference responses.
    """
    rouge = Rouge()
    scores = rouge.get_scores(generated_responses, reference_responses, avg=True)
    return scores

def evaluate_fidelity(reference_responses, generated_responses):
    """
    Evaluate the fidelity of generated responses compared to reference responses.
    Fidelity is measured as the proportion of key information present in the generated responses.
    """
    fidelity_scores = []
    for ref, gen in zip(reference_responses, generated_responses):
        ref_words = set(ref.split())
        gen_words = set(gen.split())
        common_words = ref_words.intersection(gen_words)
        fidelity = len(common_words) / len(ref_words) if len(ref_words) > 0 else 0
        fidelity_scores.append(fidelity)
    return np.mean(fidelity_scores)

def evaluate_recall(reference_responses, generated_responses):
    """
    Evaluate the recall of generated responses compared to reference responses.
    Recall is measured as the proportion of relevant information from the reference responses that is present in the generated responses.
    """
    recall_scores = []
    for ref, gen in zip(reference_responses, generated_responses):
        ref_words = set(ref.split())
        gen_words = set(gen.split())
        relevant_words = ref_words.intersection(gen_words)
        recall = len(relevant_words) / len(ref_words) if len(ref_words) > 0 else 0
        recall_scores.append(recall)
    return np.mean(recall_scores)

def evaluate_relevance(reference_responses, generated_responses):
    """
    Evaluate the relevance of generated responses compared to reference responses.
    Relevance is measured using cosine similarity between the TF-IDF vectors of the responses.
    """
    vectorizer = TfidfVectorizer().fit(reference_responses + generated_responses)
    ref_vectors = vectorizer.transform(reference_responses)
    gen_vectors = vectorizer.transform(generated_responses)
    relevance_scores = cosine_similarity(ref_vectors, gen_vectors).diagonal()
    return np.mean(relevance_scores)

def comprehensive_evaluation(reference_responses, generated_responses):
    """
    Perform a comprehensive evaluation of generated responses compared to reference responses.
    """
    accuracy = evaluate_accuracy(reference_responses, generated_responses)
    bleu_score = evaluate_bleu(reference_responses, generated_responses)
    rouge_scores = evaluate_rouge(reference_responses, generated_responses)
    fidelity = evaluate_fidelity(reference_responses, generated_responses)
    recall = evaluate_recall(reference_responses, generated_responses)
    relevance = evaluate_relevance(reference_responses, generated_responses)
    
    return {
        "accuracy": accuracy,
        "bleu_score": bleu_score,
        "rouge-1_f1": rouge_scores['rouge-1']['f'],
        "rouge-1_precision": rouge_scores['rouge-1']['p'],
        "rouge-1_recall": rouge_scores['rouge-1']['r'],
        "rouge-2_f1": rouge_scores['rouge-2']['f'],
        "rouge-2_precision": rouge_scores['rouge-2']['p'],
        "rouge-2_recall": rouge_scores['rouge-2']['r'],
        "rouge-l_f1": rouge_scores['rouge-l']['f'],
        "rouge-l_precision": rouge_scores['rouge-l']['p'],
        "rouge-l_recall": rouge_scores['rouge-l']['r'],
        "fidelity": fidelity,
        "recall": recall,
        "relevance": relevance
    }

# Fonction pour sauvegarder les résultats dans un fichier CSV

def save_results_to_csv(evaluation_results, filename="evaluation_results.csv"):
    """
    Save evaluation results to a CSV file.
    """
    # Convert the results dictionary to a DataFrame
    df = pd.DataFrame([evaluation_results])
    
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)

# Exemple d'utilisation des fonctions d'évaluation et de sauvegarde

# Supposons que nous avons des réponses de référence pour évaluer
reference_responses = [
    "The study shows a significant increase in productivity."
]

# Réponses générées par le système RAG
generated_responses = [
    "The study indicates a significant increase in productivity."
]

# Évaluation des réponses générées
evaluation_results = comprehensive_evaluation(reference_responses, generated_responses)

# Affichage des résultats d'évaluation
print("Résultats de l'évaluation :")
print(evaluation_results)

# Sauvegarde des résultats dans un fichier CSV
save_results_to_csv(evaluation_results)
