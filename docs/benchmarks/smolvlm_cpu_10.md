# Comparaison CPU — OCR v2 et SmolVLM

Date d'exécution : 21 novembre 2026

## Protocole

La comparaison utilise les dix premiers reçus du split test SROIE, déjà examinés lors de
l'ajustement des règles OCR. Cet essai sert de test fonctionnel. SmolVLM est exécuté
localement sans CUDA, avec le prompt compact et un plafond de 512 nouveaux tokens. Le
few-shot sélectionne deux exemples du train et exclut la cible et toute image de même
SHA-256. L'inspection du gabarit de conversation confirme la présence de la consigne
JSON et des réponses d'exemple.

## Résultats

| Méthode | Tests terminés | JSON syntaxiquement valides | Sorties conformes au schéma | Macro-F1 des champs | Temps moyen après prétraitement |
|---|---:|---:|---:|---:|---:|
| OCR rules v2 | 10 | 10/10 | 10/10 | 0,864 | 0,623 s |
| SmolVLM compact zero-shot | 10 | 4/10 | 0/10 | 0 | 55,672 s |
| SmolVLM compact few-shot | 1 | 0/1 | 0/1 | — | 173,216 s |

Les latences proviennent des traces du service après prétraitement. L'OCR tourne
dans Docker et SmolVLM sous Windows. Le F1 pénalise les prédictions manquantes.

Sur cet échantillon, l'OCR obtient un F1 de 0,700 pour la date, 0,876 pour l'adresse,
0,878 pour le commerçant et 1,000 pour le total. Trois documents sur dix sont entièrement
exacts.

En zero-shot, cinq réponses contiennent un objet incomplet et une présente une autre
erreur de syntaxe. Les quatre objets JSON valides utilisent une structure différente du
contrat attendu. En few-shot, le premier document produit un objet incomplet en près
de trois minutes. Le second a été interrompu pendant l'inférence. Un seul résultat
few-shot laisse la qualité de cette stratégie indéterminée.

## Décision

L'OCR rules v2 sert à préremplir les champs en vue d'une revue humaine. Les dix essais
zero-shot de SmolVLM confirment que le validateur rejette les sorties incompatibles.
L'efficacité métier de l'OCR se juge sur le benchmark complet et le temps de correction
reste à mesurer. L'évaluation Qwen et l'entraînement LoRA restent à réaliser.

Les résultats machine-readable sont dans
`artifacts/benchmarks/cpu-comparison/comparison.json`. Le rapport zero-shot complet est
dans `artifacts/benchmarks/smolvlm-cpu-zero-shot-10/report.json`, et les réponses brutes
restent associées à chaque job sous son dossier `inference/jobs`.
