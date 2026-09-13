"""
============================================================
 INFÉRENCE ONNX — SBERT UNIQUEMENT
 Le modèle de sentiment a été déplacé vers son propre service
 (voir sentiment-service/app.py) pour éviter tout risque de
 dépassement mémoire en chargeant les deux modèles ensemble.
============================================================
"""
import gc
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

SBERT_DIR = "onnx_sbert"

_sbert_tokenizer = None
_sbert_session = None


def _session_options():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    opts.enable_mem_reuse = False
    return opts


def _get_sbert():
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
