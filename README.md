# Multimodal Document Intelligence

Prototype d'extraction de données depuis des reçus et factures, avec preuves,
scores de confiance, contrôles métier et validation humaine.

## Essayer depuis un dépôt vierge

```powershell
docker compose up -d --build --wait
curl.exe -X POST http://127.0.0.1:8010/v1/documents/extract `
  -F "file=@output/pdf/invoice-demo/invoice-part-payment.pdf" `
  -F "method=ocr_rules" -F "document_id=demo-invoice"
```

Sous Windows, `./scripts/docker.ps1` remplace `docker` si le client n'apparaît pas dans
le `PATH`. Ouvrez la revue sur [localhost:7860](http://127.0.0.1:7860), chargez le
dossier et confirmez ou corrigez les valeurs. L'acompte de l'exemple
fait apparaître `total = 300.00`, `balance_due = 200.00` et `currency = EUR`.

Après la revue, téléchargez le JSON avec `GET /v1/reviews/{case_id}/export`. Le
`review_case_id` est renvoyé lors de l'extraction. La [documentation API](http://127.0.0.1:8010/docs)
permet aussi de tester les opérations d'extraction, d'attribution et de validation.

Les PDFs d'exemple sont synthétiques. Le démarrage fonctionne sans corpus, GPU ni
profil de calibration local. Consultez le [bilan des évaluations](docs/results.md).

## MVP

Le MVP traite d'abord les reçus CORD/SROIE et produit un JSON commun : commerçant,
date, devise, sous-total, taxes, total et lignes d'achat. Une facture générique est
acceptée par le contrat, mais son évaluation vient après la baseline reçus.

## Jalons

1. **Contrat et validation — terminé** — schéma Pydantic et politique de revue.
2. **Données** — adaptateurs CORD/SROIE et audit des doublons entre splits.
3. **Prétraitement — terminé** — PDF vers images, orientation et redressement.
4. **Baseline OCR — terminé** — OCR, ordre de lecture et extraction explicable.
5. **Baseline VLM** — consignes JSON et validation de la sortie. SmolVLM testé sur CPU, Qwen à évaluer.
6. **Incertitude — terminé** — calibration par champ et file de validation humaine.
7. **LoRA/PEFT** — seulement si le gain attendu et le GPU le justifient.
8. **Interface de revue — terminé** — Gradio, preuves et corrections humaines.
9. **Produit — terminé** — FastAPI, traces JSONL/MLflow, Docker et tests d'intégration.

## Démarrage local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,api,ocr,ui]"
python -m pytest --basetemp=tmp/pytest
```

Les jeux de données et modèles ne sont jamais commités. Voir `data/README.md`.

## Exécuter un benchmark reproductible

Le runner batch filtre un split, reprend les documents déjà prédits, conserve les échecs
séparément et produit un rapport qualité/revue/latence ainsi qu'un historique immuable :

```powershell
docintel-benchmark data\processed\sroie-smoke\sroie.jsonl `
  --method ocr_rules --split test --limit 10 `
  --output-dir artifacts\benchmarks\sroie-ocr-smoke
```

Pour utiliser le Tesseract inclus dans Docker et monter automatiquement les données :

```powershell
.\scripts\benchmark.ps1 -Manifest data\processed\sroie-smoke\sroie.jsonl `
  -OutputName sroie-ocr-smoke -Limit 10 -Overwrite
```

Sans `-Overwrite`, les prédictions valides existantes sont ignorées et le traitement
reprend aux documents manquants. Le smoke test est documenté dans
[`docs/benchmarks/sroie_ocr_smoke.md`](docs/benchmarks/sroie_ocr_smoke.md), et le benchmark
des 361 documents test dans
[`docs/benchmarks/sroie_ocr_full.md`](docs/benchmarks/sroie_ocr_full.md).
L'amélioration OCR v2 et le recalcul après retrait des images déjà vues et des doublons
sont décrits dans
[`docs/benchmarks/sroie_ocr_iteration_3.md`](docs/benchmarks/sroie_ocr_iteration_3.md).
L'audit du montant total et la comparaison des règles v2 et v3 figurent dans
[`docs/benchmarks/ocr_total_audit.md`](docs/benchmarks/ocr_total_audit.md).
La [démo reçus et factures](docs/benchmarks/document_flow_demo.md) vérifie le dépôt,
l'extraction, la revue et l'export JSON sur huit documents.

Pour reproduire le corpus et le benchmark complets :

```powershell
docintel-prepare sroie --output-dir data\processed\sroie-full
.\scripts\benchmark.ps1 -Manifest data\processed\sroie-full\sroie.jsonl `
  -OutputName sroie-ocr-full -Limit 361 -Overwrite
```

## Préparer un échantillon de données

```powershell
python -m pip install -e ".[data,dev]"
docintel-prepare cord_v2 --limit-per-split 10
docintel-prepare sroie --limit-per-split 10
```

Les adaptateurs produisent le même contrat JSONL et conservent un masque explicite des
champs réellement annotés par chaque source.

## Prétraiter un document

```powershell
python -m pip install -e ".[ocr,dev]"
docintel-preprocess chemin\facture.pdf
docintel-preprocess chemin\recu.jpg --max-long-edge 2400
```

Chaque page devient un PNG RGB accompagné d'un rapport JSON : empreintes SHA-256,
dimensions, transformations appliquées et signaux de qualité. Les PDF sont rendus à
200 DPI par défaut et le redressement automatique est limité aux faibles inclinaisons.

## Exécuter la baseline OCR

Le paquet Python `pytesseract` ne contient pas le moteur natif. Installez Tesseract OCR
et rendez la commande `tesseract` accessible dans `PATH`, puis lancez :

```powershell
docintel-ocr data\interim\preprocessed\mon-document\page-0001.png `
  --document-id mon-document --language eng --output artifacts\ocr\mon-document.json
```

La baseline utilise le TSV mot-à-mot, reconstruit les lignes et conserve les zones et
confiances comme preuves. Le mode de segmentation par défaut est `PSM 4`, adapté aux
documents principalement organisés en une colonne.

Les prédictions JSONL se mesurent avec le même masque de vérité terrain que les futurs
modèles VLM :

```powershell
docintel-evaluate data\processed\sroie.jsonl artifacts\ocr\predictions.jsonl `
  --output artifacts\ocr\metrics.json
```

## Exécuter la baseline VLM zero-shot

```powershell
python -m pip install -e ".[vlm]"
docintel-vlm data\interim\preprocessed\mon-document\page-0001.png `
  --document-id mon-document --model-id Qwen/Qwen3-VL-2B-Instruct
```

Deux stratégies sont disponibles et tracées séparément :

```powershell
# Schéma JSON complet, stratégie de référence
docintel-vlm page-0001.png --document-id doc-001 --prompt-style json_schema

# Gabarit compact, utile pour les petits modèles ou les essais CPU
docintel-vlm page-0001.png --document-id doc-001 --prompt-style compact
```

Le lanceur de benchmark exécute l'OCR dans Docker et les méthodes VLM dans
l'environnement Python local, où PyTorch et le cache Hugging Face sont disponibles :

```powershell
.\scripts\benchmark.ps1 -Method vlm_zero_shot -PromptStyle compact `
  -ModelId HuggingFaceTB/SmolVLM-256M-Instruct -Limit 1 `
  -OutputName smolvlm-cpu-smoke -Overwrite
```

Le backend est chargé à la demande. Chaque exécution conserve la réponse brute, la version
du prompt, son SHA-256, la latence, le périphérique, la mémoire GPU maximale éventuelle,
la validité syntaxique du JSON et la conformité au schéma Pydantic. Les poids du modèle ne
sont pas téléchargés à l'installation.

Sur les dix reçus du test fonctionnel CPU, l'OCR v2 a produit dix sorties conformes au
schéma, contre zéro pour SmolVLM-256M zero-shot. Ces reçus avaient déjà servi à ajuster
l'OCR. Le protocole et les mesures figurent dans `docs/benchmarks/smolvlm_cpu_10.md`.

Pour construire un catalogue few-shot exclusivement depuis le train puis lancer deux
exemples en contexte :

```powershell
docintel-build-fewshot data\processed\sroie.jsonl `
  --data-root data\processed --output data\processed\sroie-fewshot.jsonl

docintel-vlm page-0001.png --document-id document-001 `
  --examples data\processed\sroie-fewshot.jsonl --few-shot-count 2 `
  --target-dataset sroie
```

La sélection exclut le document cible, les images de même SHA-256 et tout exemple qui
n'appartient pas au split d'entraînement. Les identifiants sélectionnés sont enregistrés
dans la trace d'inférence.

Sur une machine GPU, l'expérience Qwen complète charge le modèle une seule fois puis
compare les trois stratégies sur le même échantillon :

```powershell
.\scripts\qwen-gpu-experiment.ps1 -Limit 10 -Overwrite
```

Le précontrôle refuse une exécution CPU accidentelle avant de télécharger les poids. Le
rapport agrégé est écrit dans `artifacts/qwen-experiment/experiment.json`. Consultez
`docs/qwen_gpu_experiment.md` pour le protocole et les artefacts produits.

Le corpus supervisé portable pour un futur fine-tuning LoRA/PEFT se génère avec
`.\scripts\export-sft.ps1`. Sa structure, les garde-fous contre la fuite du test et les
limites matérielles actuelles sont décrits dans `docs/lora_dataset.md`.

## Calibrer les confiances

La calibration refuse les splits train et test. Elle doit être exécutée séparément pour
chaque combinaison méthode/modèle/prompt :

```powershell
docintel-calibrate data\processed\validation.jsonl `
  artifacts\vlm\validation-predictions.jsonl `
  --target-accuracy 0.95 --output artifacts\vlm\calibration.json
```

Le profil contient un seuil par champ, la couverture auto-acceptée, l'accuracy obtenue,
le Brier score et l'erreur de calibration attendue. Si les données sont insuffisantes ou
si la cible de qualité n'est pas atteinte, le champ est envoyé en revue par sécurité.

La file de revue utilise SQLite. Elle conserve séparément la prédiction originale, la
correction humaine et l'historique des événements afin qu'aucune sortie modèle ne soit
écrasée.

## Lancer l'interface de revue

```powershell
python -m pip install -e ".[ui]"
docintel-review-ui --database artifacts\review\review.sqlite3
```

Ouvrez ensuite `http://127.0.0.1:7860`. L'interface affiche les pages du document, les
erreurs bloquantes, les champs et les lignes d'achat. Seules les valeurs sont modifiables ;
les confiances et textes bruts restent visibles pour l'audit.

## Lancer l'API

```powershell
python -m pip install -e ".[api,ocr,mlops]"
$env:MLFLOW_TRACKING_URI = "file:./artifacts/mlruns"
docintel-api --host 127.0.0.1 --port 8000
```

La documentation interactive est disponible sur `http://127.0.0.1:8000/docs`. L'endpoint
`POST /v1/documents/extract` reçoit un fichier multipart, une méthode optionnelle
`ocr_rules`, `vlm_zero_shot` ou `vlm_few_shot`, et un `document_id` optionnel. Les requêtes
VLM chargent le modèle à la première inférence ; le catalogue few-shot doit être injecté
dans le service avant d'utiliser cette stratégie.

Exemple OCR avec PowerShell :

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/documents/extract `
  -F "file=@chemin/recu.png" -F "method=ocr_rules" -F "document_id=recu-001"
```

Chaque exécution réussie crée un dossier immuable sous `artifacts/jobs` avec l'entrée,
les pages prétraitées et un `result.json` autonome. Elle écrit aussi une trace sans valeurs
métier dans `artifacts/traces/inference.jsonl`, puis envoie automatiquement les sorties
incertaines vers SQLite. Si `MLFLOW_TRACKING_URI` est défini, les paramètres, métriques et
artefacts JSON sont également journalisés dans MLflow. Une indisponibilité du serveur de
traces n'interrompt pas l'extraction.

Le profil de calibration est optionnel. Dans Compose, `DOCINTEL_CALIBRATION_PROFILE`
peut désigner un profil existant sous `/app/calibration`. Si aucun profil compatible
n'est configuré, la pile locale envoie les documents en revue. Le profil OCR factures
est actif dans la pile Docker. Il démarque le total et le solde et maintient une revue
humaine sur toutes les factures. `DOCINTEL_OCR_INVOICE_RULES=false` force la référence
OCR v2 pour les expériences de reproduction.

L'endpoint `GET /v1/reviews/{case_id}/export` répond `409` si la revue est ouverte. Le
JSON d'une revue fermée contient son statut et l'extraction relue. Il exporte la
correction lorsqu'elle existe et conserve la prédiction originale dans l'historique.

## Lancer la pile Docker

```powershell
docker compose up --build
```

Si un terminal isolé ne résout pas la commande `docker`, utilisez le lanceur fourni :

```powershell
.\scripts\docker.ps1 compose up --build
```

Ce lanceur détecte notamment l'installation utilisateur de Docker Desktop utilisée sur
Windows et évite de confondre un problème de `PATH` avec une absence de Docker.

La pile expose par défaut l'API sur `http://127.0.0.1:8010`, la revue Gradio sur
`http://127.0.0.1:7860` et MLflow sur `http://127.0.0.1:5000`. L'image CPU installe
Tesseract anglais/français. Les poids Qwen et le runtime GPU ne sont volontairement pas
intégrés à l'image : ils doivent être ajoutés dans une image GPU dédiée ou montés depuis
un cache Hugging Face.

Le serveur MLflow autorise explicitement le nom interne Docker `mlflow:5000` ainsi que
les accès locaux, tout en conservant sa protection contre les en-têtes d'hôte inattendus.

Les ports de l'API et de l'interface peuvent être remplacés sans modifier le fichier :

```powershell
$env:DOCINTEL_API_PORT = "8020"
$env:DOCINTEL_UI_PORT = "7870"
.\scripts\docker.ps1 compose up -d
```
