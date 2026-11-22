# Benchmark OCR SROIE — itération 3

Date d'exécution : 21 novembre 2026  
Run test v2 : `96c1405e-5ee6-48a6-ad97-50d50612fa71`

## Objectif

Améliorer le total et le commerçant. Les règles de cette itération ont été choisies sur
les 99 reçus de validation. Les règles héritées de l'itération 2 ont déjà été ajustées
sur dix reçus du split test. Le snapshot contient aussi des images identiques entre splits.

## Modifications

- reconnaissance des décimales OCR séparées ou confondues avec deux-points, par exemple
  `65. 20` et `92:80` ;
- exclusion des lignes `TOTAL QUANTITY`, `TOTAL POINTS`, `ACCEPTED TOTAL`, des totaux
  d'items et des récapitulatifs GST inclus dans le total ;
- prise en charge de `SUB-TOTAL` ;
- recherche du commerçant dans toute la zone supérieure au lieu des huit premières
  lignes seulement ;
- bonus étendu aux suffixes et secteurs commerciaux, avec pénalité pour les lignes
  ressemblant à une adresse ;
- version d'inférence portée à `ocr-rules-v2`.

## Résultats validation

| Champ | F1 v1 | F1 v2 | Delta |
|---|---:|---:|---:|
| Total | 0,47 | 0,57 | +0,10 |
| Commerçant | 0,64 | 0,73 | +0,09 |
| Adresse | 0,72 | 0,77 | +0,05 |
| Date | 0,78 | 0,78 | 0,00 |

Le rappel de détection des documents erronés passe de 45 % à 60 % sur validation.

## Résultats sur le split test d'origine

| Champ | Exact v1 | Exact v2 | F1 v1 | F1 v2 | Delta F1 |
|---|---:|---:|---:|---:|---:|
| Total | 46,81 % | 51,80 % | 0,503 | 0,566 | +0,063 |
| Commerçant | 37,95 % | 38,50 % | 0,662 | 0,698 | +0,036 |
| Adresse | 30,47 % | 30,75 % | 0,727 | 0,736 | +0,009 |
| Date | 65,53 % | 65,53 % | 0,752 | 0,752 | 0,000 |

| Indicateur sélectif | v1 | v2 |
|---|---:|---:|
| Revue avant calibration | 34,90 % | 47,65 % |
| Documents entièrement exacts | 9,97 % | 10,53 % |
| Exactitude des auto-acceptations | 14,04 % | 17,99 % |
| Précision de la revue | 97,62 % | 97,67 % |
| Rappel de détection des erreurs | 37,85 % | 52,01 % |
| Latence moyenne | 692 ms | 614 ms |

Les 361 documents sont traités avec 100 % de JSON valide et zéro échec. Le temps mur du
run v2 est de 10 min 39 s.

## Effet des recoupements

Le snapshot contient 16 groupes d'images identiques, dont 9 répartis entre plusieurs
splits. Huit reçus test partagent une image avec le train ou la validation. Le recalcul
exclut ces images, les dix reçus déjà examinés lors de l'itération 2 et leurs copies,
puis conserve un seul reçu par image. Il reste 340 reçus test.

| Indicateur | OCR v1 | OCR v2 |
|---|---:|---:|
| Documents entièrement exacts | 33/340 (9,7 %) | 35/340 (10,3 %) |
| Totaux exacts | 153/340 (45,0 %) | 172/340 (50,6 %) |
| Macro-F1 des champs | 0,654 | 0,682 |

Ce recalcul réutilise les prédictions archivées. Il mesure la sensibilité des scores
aux recoupements connus. Une nouvelle évaluation indépendante exige un jeu d'images
réservé avant d'ajuster les règles et les seuils.

## Calibration à 95 %

Le profil v2 est appris uniquement sur validation. La date atteint la cible avec un seuil
de 0,789. Le commerçant, l'adresse et le total restent `target_not_reached`, malgré le
passage du total de 41 à 49 prédictions validation exactes.

La politique sûre continue donc d'envoyer 100 % des documents test en revue, avec un
rappel des erreurs de 100 % et une précision de revue de 89,47 %. Cette itération améliore
la baseline, mais ne justifie toujours pas l'auto-acceptation métier.

## Artefacts

- `artifacts/benchmarks/sroie-ocr-validation-v2/report.json`
- `artifacts/benchmarks/sroie-ocr-full-v2/report.json`
- `artifacts/benchmarks/sroie-ocr-full-v2/calibrated-metrics-95.json`
- `artifacts/calibration/sroie-ocr-rules-v2-95.json`
