# ============================================================
#  IMPORTS
#  ⚠️ torch, sentence-transformers et transformers.pipeline supprimés.
#     Tout le reste (Mongo, Flask, logique métier) est identique à l'original.
# ============================================================
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson import ObjectId
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random

from onnx_models import encode_texts  # <-- remplace SentenceTransformer (SBERT uniquement, plus de sentiment ici)


# ============================================================
#  CONFIGURATION INITIALE
# ============================================================
load_dotenv()
app = Flask(__name__)

MONGO_URI = os.getenv('MONGO_URI')
client = MongoClient(MONGO_URI)
db = client['test']

print("✅ Modèles ONNX chargés (voir onnx_models.py)")


# ============================================================
#  ROUTE HEALTH CHECK — pour UptimeRobot / monitoring
#  Ne charge aucun modèle, répond toujours 200 rapidement.
# ============================================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200


# ============================================================
#  BRIQUE 1 : CONTENT-BASED (SBERT via ONNX)
# ============================================================
def score_content_based(description_etudiant, courses):
    """Calcule le score SBERT entre l'étudiant et chaque cours."""
    vecteurs_courses = np.array([c['embedding'] for c in courses])
    vecteur_etudiant = encode_texts([description_etudiant])[0]
    scores = cosine_similarity([vecteur_etudiant], vecteurs_courses)[0]
    return {c['_id']: float(scores[i]) for i, c in enumerate(courses)}, vecteur_etudiant


# ============================================================
#  BRIQUE 2 : FEEDBACK (note + sentiment + popularité) — inchangé
# ============================================================
def score_feedback(course_ids):
    """Agrège note, sentiment et popularité par cours depuis MongoDB."""
    pipeline_feedback = [
        {'$match': {'course': {'$in': course_ids}}},
        {'$group': {
            '_id': '$course',
            'note_moyenne': {'$avg': '$rating'},
            'sentiment_moyen': {'$avg': '$sentiment_score'},
            'nb_avis': {'$sum': 1}
        }}
    ]
    stats = {s['_id']: s for s in db['feedbacks'].aggregate(pipeline_feedback)}
    nb_avis_list = [stats.get(cid, {}).get('nb_avis', 0) for cid in course_ids]
    max_log_avis = np.log1p(max(nb_avis_list)) if max(nb_avis_list) > 0 else 1

    resultat = {}
    for cid in course_ids:
        s = stats.get(cid, {})
        resultat[cid] = {
            'note_norm': s.get('note_moyenne', 2.5) / 5.0,
            'sentiment_norm': (s.get('sentiment_moyen', 0) + 1) / 2.0,
            'popularite_norm': np.log1p(s.get('nb_avis', 0)) / max_log_avis if max_log_avis > 0 else 0,
            'nb_avis': s.get('nb_avis', 0)
        }
    return resultat


# ============================================================
#  BRIQUE 3 : COLLABORATIF (étudiants similaires, via ONNX)
# ============================================================
def mode(liste):
    return max(set(liste), key=liste.count) if liste else None


def score_collaboratif(vecteur_etudiant, language, level, user_id=None, max_voisins_potentiels=30):
    """Calcule le score CF pour chaque cours à partir des voisins (Enrollment payés)."""
    pipeline_profils = [
        {'$match': {'status': 'paid'}},
        {'$lookup': {
            'from': 'courses', 'localField': 'course',
            'foreignField': '_id', 'as': 'course_info'
        }},
        {'$unwind': '$course_info'},
        {'$group': {
            '_id': '$user',
            'languages': {'$push': '$course_info.language'},
            'levels': {'$push': '$course_info.level'},
            'descriptions': {'$push': '$course_info.description'},
            'courses_suivis': {'$push': '$course'}
        }}
    ]
    profils_bruts = list(db['enrollments'].aggregate(pipeline_profils))

    profils = [{
        'user_id': p['_id'],
        'language': mode(p['languages']),
        'level': mode(p['levels']),
        'description_profil': ' | '.join(p['descriptions'])[:1000],
        'courses_suivis': p['courses_suivis']
    } for p in profils_bruts]

    voisins_potentiels = [p for p in profils if p['language'] == language and p['level'] == level]

    # ⚠️ Limite le nombre de textes envoyés à l'encodage batch pour éviter le pic RAM
    if len(voisins_potentiels) > max_voisins_potentiels:
        voisins_potentiels = random.sample(voisins_potentiels, max_voisins_potentiels)

    score_cf_dict = defaultdict(float)
    voisins = []
    if voisins_potentiels:
        vecteurs_voisins = encode_texts([p['description_profil'] for p in voisins_potentiels])
        sims = cosine_similarity([vecteur_etudiant], vecteurs_voisins)[0]
        for p, sim in zip(voisins_potentiels, sims):
            p['similarite'] = float(sim)

        voisins = sorted(voisins_potentiels, key=lambda x: x['similarite'], reverse=True)[:10]

        for voisin in voisins:
            for cid in voisin['courses_suivis']:
                score_cf_dict[cid] += voisin['similarite']

        if user_id:
            deja_suivi = next((p['courses_suivis'] for p in profils
                                if str(p['user_id']) == user_id), [])
            for cid in deja_suivi:
                score_cf_dict.pop(cid, None)

    return score_cf_dict, voisins

# ============================================================
#  BRIQUE 4 : ASSEMBLAGE FINAL — inchangé
# ============================================================
def assembler_scores(courses, scores_bert, feedback_stats, score_cf_dict=None):
    max_cf = max(score_cf_dict.values()) if score_cf_dict else 0

    for c in courses:
        cid = c['_id']
        fb = feedback_stats[cid]
        c['score_bert'] = round(scores_bert[cid], 4)
        c['note_norm'] = round(fb['note_norm'], 3)
        c['sentiment_norm'] = round(fb['sentiment_norm'], 3)
        c['nb_avis'] = fb['nb_avis']

        if score_cf_dict is not None:
            brut = score_cf_dict.get(cid)
            score_cf = (brut / max_cf) if brut and max_cf > 0 else 0.5
            c['score_cf'] = round(score_cf, 3)
            c['score_final'] = round(
                0.45 * scores_bert[cid]
              + 0.25 * score_cf
              + 0.10 * fb['note_norm']
              + 0.10 * fb['sentiment_norm']
              + 0.10 * fb['popularite_norm'], 4
            )
        else:
            c['score_final'] = round(
                0.55 * scores_bert[cid]
              + 0.15 * fb['note_norm']
              + 0.15 * fb['sentiment_norm']
              + 0.15 * fb['popularite_norm'], 4
            )

        c['_id'] = str(c['_id'])
        c['teacher'] = str(c.get('teacher', ''))
        del c['embedding']

    return sorted(courses, key=lambda x: x['score_final'], reverse=True)[:3]


# ============================================================
#  ROUTE 1 : ENCODER UNE FORMATION (via ONNX)
# ============================================================
@app.route('/encoder-formation', methods=['POST'])
def encoder_formation():
    data = request.get_json()
    description = data.get('description', '')
    embedding = encode_texts([description])[0].tolist()
    return jsonify({'success': True, 'embedding': embedding})


# ============================================================
#  ROUTE 2 : ANALYSER LE SENTIMENT
#  ⚠️ Deplacee vers le service dedie lingia-sentiment (voir
#     sentiment-service/app.py) pour eviter de charger SBERT et le
#     modele de sentiment dans le meme processus (risque OOM).
# ============================================================


# ============================================================
#  ROUTE 3 : RECOMMANDER — inchangée dans sa logique
# ============================================================
@app.route('/recommander', methods=['POST'])
def recommander():
    data = request.get_json()
    language, level = data['language'], data['level']
    description_etudiant = data['description']

    courses = list(db['courses'].find({'language': language, 'level': level}))
    courses = [c for c in courses if c.get('embedding')]
    if not courses:
        return jsonify({'success': False, 'error': 'Aucun cours trouvé'}), 404

    scores_bert, _ = score_content_based(description_etudiant, courses)
    feedback_stats = score_feedback([c['_id'] for c in courses])

    top = assembler_scores(courses, scores_bert, feedback_stats, score_cf_dict=None)
    return jsonify({'success': True, 'recommendations': top})


# ============================================================
#  ROUTE 4 : RECOMMANDER HYBRIDE — inchangée dans sa logique
# ============================================================
@app.route('/recommander-hybride', methods=['POST'])
def recommander_hybride():
    data = request.get_json()
    language, level = data['language'], data['level']
    description_etudiant = data['description']
    user_id = data.get('user_id')

    courses = list(db['courses'].find({'language': language, 'level': level}))
    courses = [c for c in courses if c.get('embedding')]
    if not courses:
        return jsonify({'success': False, 'error': 'Aucun cours trouvé'}), 404

    scores_bert, vecteur_etudiant = score_content_based(description_etudiant, courses)
    feedback_stats = score_feedback([c['_id'] for c in courses])
    score_cf_dict, _ = score_collaboratif(vecteur_etudiant, language, level, user_id)

    top = assembler_scores(courses, scores_bert, feedback_stats, score_cf_dict=score_cf_dict)
    return jsonify({'success': True, 'recommendations': top})


# ============================================================
#  ROUTE 5 (DEBUG) : tester le collaboratif isolément
# ============================================================
@app.route('/debug-cf', methods=['POST'])
def debug_cf():
    data = request.get_json()
    language, level = data['language'], data['level']
    description_etudiant = data['description']
    user_id = data.get('user_id')

    vecteur_etudiant = encode_texts([description_etudiant])[0]
    score_cf_dict, voisins = score_collaboratif(vecteur_etudiant, language, level, user_id)

    if not voisins:
        return jsonify({'success': True, 'message': 'Aucun voisin trouvé', 'voisins': []})

    voisins_info = [{
        'user_id': str(v['user_id']),
        'similarite': round(v['similarite'], 3),
        'nb_cours_suivis': len(v['courses_suivis'])
    } for v in voisins]

    return jsonify({
        'success': True,
        'nb_voisins': len(voisins),
        'voisins': voisins_info,
        'score_cf_par_cours': {str(k): round(v, 3) for k, v in score_cf_dict.items()}
    })


# ============================================================
#  LANCEMENT DU SERVEUR
# ============================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5001)))
