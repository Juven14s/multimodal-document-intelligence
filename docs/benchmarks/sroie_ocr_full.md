# Benchmark OCR SROIE — corpus test complet

Date d'exécution : 21 novembre 2026  
Run d'inférence : `c6c84002-2d6a-4e1e-8619-768af62472bc`

## Protocole

- Source : `jsdnrs/ICDAR2019-SROIE` via Hugging Face Datasets.
- Snapshot préparé : 987 documents et 821 Mo d'images PNG.
- Splits de benchmark : 527 train, 99 validation et 361 test.
- Moteur : Tesseract 5.3.0 dans l'image Docker CPU.
- Champs évalués : commerçant, adresse, date et total, selon le masque canonique.
- Calibration : ajustée exclusivement sur les 99 documents validation.
- Coût d'API externe : 0 USD.

La préparation filtre les régions OCR aux boîtes dégénérées et exclut du masque les
annotations absentes ou non normalisables. Aucun document n'est supprimé pour ces motifs.

## Résultats d'extraction

| Indicateur | Résultat |
|---|---:|
| Documents traités | 361 / 361 |
| Échecs d'inférence | 0 |
| Couverture | 100 % |
| JSON valide | 100 % |
| Revue avec la politique initiale | 34,90 % |
| Latence moyenne | 692 ms |
| Latence P50 | 672 ms |
| Latence P95 | 1 109 ms |
| Temps mur initial | 11 min 51,6 s |

| Champ | Support | Prédit | Exact match | F1 |
|---|---:|---:|---:|---:|
| Date | 351 | 261 | 65,53 % | 0,752 |
| Adresse | 361 | 305 | 30,47 % | 0,727 |
| Commerçant | 361 | 345 | 37,95 % | 0,662 |
| Total | 361 | 311 | 46,81 % | 0,503 |

Le smoke test de 10 documents surestimait fortement la généralisation, notamment pour le
total, dont le F1 passait artificiellement à 1,00 sur ce petit échantillon.

## Qualité du routage humain

Un document est considéré exact seulement si tous ses champs annotés correspondent.

| Indicateur | Politique initiale |
|---|---:|
| Documents entièrement exacts | 36 / 361 (9,97 %) |
| Documents auto-acceptés | 235 |
| Exactitude des auto-acceptations | 14,04 % |
| Documents envoyés en revue | 126 |
| Précision de la revue | 97,62 % |
| Rappel de détection des documents erronés | 37,85 % |

La politique initiale signale presque toujours un vrai problème lorsqu'elle demande une
revue, mais laisse passer trop d'erreurs. Elle ne doit pas être utilisée pour une
auto-acceptation métier en l'état.

## Calibration sur le split de validation

Un profil visant 95 % de précision par champ a été appris sur validation. La date atteint
la cible avec un seuil de 0,789 et une couverture validation de 44,68 %. L'adresse, le
commerçant et le total n'atteignent pas la cible, même avec un seuil de 1,0.

La politique sûre obtenue route donc les 361 documents test en revue : rappel des erreurs
de 100 %, précision de revue de 90,03 %, aucune auto-acceptation. Des essais exploratoires
à 90 %, 80 % et 70 % ne rendent toujours pas le commerçant ni l'adresse calibrables.

## Décision

La baseline OCR est rapide, déterministe et robuste opérationnellement, mais insuffisante
pour une automatisation sans supervision. Les prochaines améliorations doivent cibler la
sélection du total et l'identification du commerçant, puis être comparées au VLM sur ce
même protocole. La calibration 95 % reste la politique de référence tant qu'une nouvelle
approche n'offre pas une couverture auto-acceptée mesurable.

L'itération suivante et le recalcul après retrait des images déjà vues et des doublons
sont documentés dans `docs/benchmarks/sroie_ocr_iteration_3.md`.

## Artefacts

- `data/processed/sroie-full/sroie.jsonl`
- `data/processed/sroie-full/sroie_audit.json`
- `artifacts/benchmarks/sroie-ocr-full/report.json`
- `artifacts/benchmarks/sroie-ocr-full/calibrated-metrics-95.json`
- `artifacts/benchmarks/sroie-ocr-validation/report.json`
- `artifacts/calibration/sroie-ocr-rules-95.json`
