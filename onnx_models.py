"""
============================================================
 INFÉRENCE ONNX — remplace torch + sentence-transformers + transformers.pipeline
 Mêmes modèles, mêmes résultats (à l'arrondi de la quantification près)

 v2 : chargement PARESSEUX (lazy) + options mémoire réduites pour
      onnxruntime, afin de tenir sous la limite de 512 Mo de Render Free.
      - Les modèles ne sont chargés qu'au premier appel qui en a besoin,
        pas tous les deux au démarrage du process.
      - Les sessions onnxruntime sont configurées en mode mono-thread,
        sans arène mémoire, ce qui réduit nettement la RAM consommée
        (l'arène mémoire par défaut pré-alloue des blocs qu'onnxruntime
        garde en réserve même une fois le calcul terminé).
============================================================
"""
import gc
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

SBERT_DIR = "onnx_sbert"
SENTIMENT_DIR = "onnx_sentiment"

# ── Caches globaux, remplis à la demande (lazy loading) ──
_sbert_tokenizer = None
_sbert_session = None
_sentiment_tokenizer = None
_sentiment_session = None


def _session_options():
    """Options onnxruntime réduisant la mémoire résidente (au prix d'un peu de vitesse)."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_cpu_mem_arena = False   # pas de pré-allocation mémoire gardée en réserve
    opts.enable_mem_pattern = False
    opts.enable_mem_reuse = False
    return opts


def _get_sbert():
    """Charge SBERT au premier appel seulement, puis réutilise la même session."""
    global _sbert_tokenizer, _sbert_session
    if _sbert_session is None:
        print("Chargement à la demande de SBERT (ONNX)...")
        _sbert_tokenizer = AutoTokenizer.from_pretrained(SBERT_DIR)
        _sbert_session = ort.InferenceSession(
            f"{SBERT_DIR}/model_quantized.onnx",
            sess_options=_session_options(),
            providers=["CPUExecutionProvider"],
        )
        print("✅ SBERT (ONNX) chargé")
    return _sbert_tokenizer, _sbert_session


def _get_sentiment():
    """Charge le modèle de sentiment au premier appel seulement."""
    global _sentiment_tokenizer, _sentiment_session
    if _sentiment_session is None:
        print("Chargement à la demande du modèle de sentiment (ONNX)...")
        _sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_DIR)
        _sentiment_session = ort.InferenceSession(
            f"{SENTIMENT_DIR}/model_quantized.onnx",
            sess_options=_session_options(),
            providers=["CPUExecutionProvider"],
        )
        print("✅ Modèle de sentiment (ONNX) chargé")
    return _sentiment_tokenizer, _sentiment_session


def _mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(np.float32)
    summed = np.sum(token_embeddings * mask, axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def encode_texts(texts):
    """Remplace sbert_model.encode(texts)."""
    tokenizer, session = _get_sbert()
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=256, return_tensors="np"
    )
    input_ids = inputs["input_ids"].astype(np.int64)
    onnx_inputs = {
        "input_ids": input_ids,
        "attention_mask": inputs["attention_mask"].astype(np.int64),
        "token_type_ids": inputs.get("token_type_ids", np.zeros_like(input_ids)).astype(np.int64),
    }
    outputs = session.run(None, onnx_inputs)
    token_embeddings = outputs[0]
    result = _mean_pooling(token_embeddings, inputs["attention_mask"])
    del outputs, token_embeddings
    gc.collect()
    return result


def analyze_sentiment(comment: str) -> int:
    """Remplace sentiment_analyzer(comment). Retourne -1, 0 ou 1."""
    if not comment.strip():
        return 0

    tokenizer, session = _get_sentiment()
    inputs = tokenizer(
        comment[:512], truncation=True, max_length=512, return_tensors="np"
    )
    input_ids = inputs["input_ids"].astype(np.int64)
    onnx_inputs = {
        "input_ids": input_ids,
        "attention_mask": inputs["attention_mask"].astype(np.int64),
        "token_type_ids": inputs.get("token_type_ids", np.zeros_like(input_ids)).astype(np.int64),
    }
    logits = session.run(None, onnx_inputs)[0]
    etoiles = int(np.argmax(logits, axis=-1)[0]) + 1
    del logits
    gc.collect()

    if etoiles >= 4:
        return 1
    elif etoiles == 3:
        return 0
    else:
        return -1
