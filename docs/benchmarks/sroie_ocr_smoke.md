# Benchmark OCR SROIE — smoke test

Date d'exécution : 20 novembre 2026  
Run : `a57b500c-c4e0-45ea-bff5-561579b1cb31`

## Protocole

- Source : `jsdnrs/ICDAR2019-SROIE` via Hugging Face Datasets.
- Échantillon : les 10 premiers documents du split test officiel.
- Moteur : Tesseract 5.3.0, anglais, PSM 4, dans l'image Docker CPU.
- Pipeline : prétraitement standard, OCR TSV, reconstruction des lignes, règles
  déterministes, validation métier et politique de revue.
- Masque SROIE : commerçant, adresse, date et total.
- Coût d'API externe : 0 USD ; le coût d'infrastructure locale n'est pas estimé.

Ce passage est un test de fonctionnement et non une estimation statistique finale. Le
snapshot préparé le 21 novembre 2026 contient 361 documents test ; son benchmark complet
est décrit dans `docs/benchmarks/sroie_ocr_full.md`.

## Résultats

| Indicateur | Résultat |
|---|---:|
| Documents traités | 10 / 10 |
| Échecs d'inférence | 0 |
| Couverture des prédictions | 100 % |
| JSON valide | 100 % |
| Recours à la revue humaine | 60 % |
| Latence OCR moyenne | 603 ms |
| Latence OCR P50 | 520 ms |
| Latence OCR P95 | 1 276 ms |
| Temps mur total | 16,01 s |

| Champ | Exact match | F1 |
|---|---:|---:|
| Date | 70 % | 0,70 |
| Commerçant | 50 % | 0,65 |
| Total | 40 % | 0,44 |
| Adresse | 0 % | 0,00 |

## Analyse d'erreurs

Six documents ont été routés vers la revue. Les problèmes observés sont :

- champ obligatoire absent : 2 occurrences ;
- confiance de classification insuffisante : 3 ;
- confiance de champ insuffisante : 1 ;
- total arithmétiquement incohérent : 2 ;
- somme des lignes incohérente : 3.

La baseline ne comporte pas encore de règle d'adresse ; le score nul sur ce champ est
donc attendu. Les priorités suivantes sont d'ajouter une extraction d'adresse explicable,
d'améliorer la sélection du total lorsqu'un reçu contient plusieurs montants récapitulatifs,
puis d'exécuter le protocole complet avant la comparaison VLM.

Les artefacts détaillés se trouvent dans `artifacts/benchmarks/sroie-ocr-smoke/` :
prédictions JSONL, échecs, traces d'inférence, rapport courant et historique des runs.
