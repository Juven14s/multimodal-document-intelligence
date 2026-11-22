# Benchmark OCR SROIE — itération 2

Date d'exécution : 20 novembre 2026  
Run final : `d00e4dfa-1d5a-404b-be97-cf44079d8fce`

## Objectif

Cette itération corrige les erreurs observées lors du smoke test initial, sans changer
l'échantillon ni le moteur OCR. Elle mesure donc l'effet des règles d'extraction, et non
un changement de données ou de modèle.

## Protocole

- mêmes 10 premiers documents du split test officiel SROIE ;
- Tesseract 5.3.0, anglais, PSM 4, image Docker CPU ;
- même prétraitement, même politique de validation et mêmes métriques ;
- coût d'API externe : 0 USD.

Le jeu de 10 documents reste un smoke test. Il ne constitue pas une estimation
statistique des performances sur l'intégralité de SROIE.

## Résultats comparés

| Indicateur | Baseline | Itération 2 | Delta |
|---|---:|---:|---:|
| Documents traités | 10 / 10 | 10 / 10 | — |
| Échecs d'inférence | 0 | 0 | — |
| JSON valide | 100 % | 100 % | 0 pt |
| Revue humaine | 60 % | 40 % | -20 pts |
| Latence OCR moyenne | 603 ms | 681 ms | +78 ms |
| Latence OCR P95 | 1 276 ms | 1 666 ms | +390 ms |
| Temps mur total | 16,01 s | 17,25 s | +1,24 s |

| Champ | EM baseline | EM itération 2 | F1 baseline | F1 itération 2 |
|---|---:|---:|---:|---:|
| Date | 70 % | 70 % | 0,70 | 0,70 |
| Commerçant | 50 % | 70 % | 0,65 | 0,88 |
| Total | 40 % | 100 % | 0,44 | 1,00 |
| Adresse | 0 % | 40 % | 0,00 | 0,88 |

La variation de latence est principalement liée à l'OCR et au faible nombre de
documents. Les règles ajoutées sont déterministes et n'appellent aucun service externe.

## Modifications évaluées

- extraction d'adresse ancrée après le commerçant, avec arrêt sur les métadonnées
  fiscales, téléphoniques et documentaires ;
- préférence pour les raisons sociales explicites dans la zone d'en-tête ;
- exclusion des faux totaux tels que `Total GST` et `Total item(s)` ;
- reconnaissance des variantes `Total (RM)`, `TakeOut Total`, `TOT` et des libellés
  OCR tronqués ;
- calcul explicable du total après ajustement d'arrondi ;
- récupération spatiale prudente lorsque Tesseract sépare le libellé d'arrondi et le
  montant final ;
- classification directe de `Tax Invoice` avec une confiance suffisante.

## Incertitude et erreurs restantes

Quatre documents sur dix sont encore envoyés en revue humaine. Les alertes restantes
sont : trois confiances de champ faibles, trois incohérences entre sous-total, taxe et
total, et trois incohérences entre lignes et sous-total. Ces alertes sont conservées :
elles signalent des limites de l'OCR ou du parsing des lignes plutôt que des échecs de
production du JSON.

Les artefacts reproductibles se trouvent dans
`artifacts/benchmarks/sroie-ocr-improved/` : prédictions JSONL, fichier d'échecs,
traces, rapport courant et historique des exécutions.
