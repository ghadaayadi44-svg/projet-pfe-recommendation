"""
============================================================
 INFÉRENCE ONNX — remplace torch + sentence-transformers + transformers.pipeline
 Mêmes modèles, mêmes résultats (à l'arrondi de la quantification près)
============================================================
"""
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# ── Chemins vers les modèles exportés (voir export_to_onnx.py) ──
SBERT_DIR = "onnx_sbert"
SENTIMENT_DIR = "onnx_sentiment"

print("Chargement du tokenizer + session ONNX pour SBERT...")
sbert_tokenizer = AutoTokenizer.from_pretrained(SBERT_DIR)
sbert_session = ort.InferenceSession(f"{SBERT_DIR}/model_quantized.onnx")
print("✅ SBERT (ONNX) chargé")

print("Chargement du tokenizer + session ONNX pour le sentiment...")
sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_DIR)
sentiment_session = ort.InferenceSession(f"{SENTIMENT_DIR}/model_quantized.onnx")
print("✅ Modèle de sentiment (ONNX) chargé")


def _mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Reproduit exactement le pooling utilisé par SentenceTransformer.encode() :
    moyenne des embeddings de tokens, pondérée par le masque d'attention
    (pour ignorer le padding).
    """
    mask = attention_mask[..., None].astype(np.float32)          # (batch, seq_len, 1)
    summed = np.sum(token_embeddings * mask, axis=1)              # (batch, hidden)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)    # évite division par 0
    return summed / counts


def encode_texts(texts):
    """
    Remplace sbert_model.encode(texts).
    Retourne un np.array de shape (nb_textes, hidden_size), identique
    en usage à ce que renvoyait sentence-transformers.
    """
    inputs = sbert_tokenizer(
        texts, padding=True, truncation=True, max_length=256, return_tensors="np"
    )
    input_ids = inputs["input_ids"].astype(np.int64)
    onnx_inputs = {
        "input_ids": input_ids,
        "attention_mask": inputs["attention_mask"].astype(np.int64),
        # Le modèle ONNX exporté exige token_type_ids même si XLM-RoBERTa
        # ne s'en sert pas réellement : on fournit des zéros (segment unique).
        "token_type_ids": inputs.get("token_type_ids", np.zeros_like(input_ids)).astype(np.int64),
    }

    outputs = sbert_session.run(None, onnx_inputs)
    token_embeddings = outputs[0]  # (batch, seq_len, hidden)
    return _mean_pooling(token_embeddings, inputs["attention_mask"])


def analyze_sentiment(comment: str) -> int:
    """
    Remplace sentiment_analyzer(comment). Retourne -1, 0 ou 1,
    exactement comme avant :
        étoiles >= 4 -> 1
        étoiles == 3 -> 0
        étoiles <= 2 -> -1
    """
    if not comment.strip():
        return 0

    inputs = sentiment_tokenizer(
        comment[:512], truncation=True, max_length=512, return_tensors="np"
    )
    input_ids = inputs["input_ids"].astype(np.int64)
    onnx_inputs = {
        "input_ids": input_ids,
        "attention_mask": inputs["attention_mask"].astype(np.int64),
        "token_type_ids": inputs.get("token_type_ids", np.zeros_like(input_ids)).astype(np.int64),
    }

    logits = sentiment_session.run(None, onnx_inputs)[0]  # (1, 5)
    etoiles = int(np.argmax(logits, axis=-1)[0]) + 1        # labels 0-4 -> étoiles 1-5

    if etoiles >= 4:
        return 1
    elif etoiles == 3:
        return 0
    else:
        return -1
