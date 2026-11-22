# Démonstration de bout en bout sur reçus et factures

## Lot et parcours

Quatre reçus publics SROIE et quatre factures synthétiques passent par l'endpoint
multipart d'extraction avec Tesseract. Les PDF couvrent une facture standard, une
facture avec acompte, une facture raster sans couche texte et une facture sur deux
pages. Les cinq pages PDF ont été rendues avec Poppler et vérifiées visuellement.

La démo utilise une instance FastAPI en conteneur et une base de revue SQLite isolée.
Pour chaque dossier, elle vérifie la présence des images dans la vue de revue, transmet
les valeurs de référence par le même constructeur de correction que l'interface, puis
relit le JSON final avec le schéma Pydantic. L'historique conserve la prédiction OCR,
la correction et les événements d'ouverture, d'attribution et de résolution.

## Résultats

| Lot | Documents en revue | Champs exacts avant revue | Faux | Manquants | Champs exacts après correction de test |
|---|---:|---:|---:|---:|---:|
| Factures synthétiques | 4/4 | 30/36 | 2 | 4 | 36/36 |
| Reçus SROIE | 4/4 | 7/16 | 9 | 0 | 16/16 |

Les huit JSON finaux respectent le schéma. Les factures sont toutes classées comme
factures. L'OCR trouve trois totaux de facture sur quatre. Les quatre dates d'échéance
sont absentes et aucun des quatre tableaux d'articles ne correspond intégralement à
la référence avant revue. Le lot vérifie le fonctionnement du parcours. Les résultats
d'extraction restent propres à ces exemples et la correction utilise les références
connues de façon automatisée.

## Cas de la facture avec acompte

L'OCR choisit **200,00** sur la ligne `Balance due`, alors que le total de facture est
**300,00 EUR**. Il propose aussi `MYR` pour la devise, avant correction vers `EUR`.
Le JSON corrigé indique la méthode `human`. Le texte OCR `Balance due 200.00` reste
dans la prédiction d'origine et dans l'historique. La valeur corrigée ne porte plus
cette ligne comme preuve.

Les fichiers `initial.json`, `final.json` et `review.json` du cas se trouvent dans le
dossier local `cases/invoice-part-payment/` de la démo. Le fichier `report.json` à la
racine du dossier de sortie résume le lot.

## Reproduire

Les factures et leurs références sont dans `output/pdf/invoice-demo`. Le script
`scripts/generate_invoice_demo.py` les régénère avec ReportLab et PyMuPDF. Le manifeste
SROIE complet est un prérequis pour les quatre reçus.

Depuis la racine du projet, utiliser un nouveau dossier de sortie à chaque passage.

```powershell
.\scripts\docker.ps1 compose run --rm --no-deps --pull never `
  -e PYTHONPATH=/app/src -v "${PWD}/src:/app/src:ro" `
  -v "${PWD}/scripts:/app/demo-scripts:ro" `
  -v "${PWD}/output/pdf/invoice-demo:/app/demo-input:ro" `
  -v "${PWD}/data:/app/data:ro" -v "${PWD}/artifacts:/app/host-artifacts" `
  api python /app/demo-scripts/document_flow_demo.py `
  --invoices /app/demo-input --sroie /app/data/processed/sroie-full/sroie.jsonl `
  --calibration /app/host-artifacts/calibration/sroie-ocr-rules-v2-95.json `
  --output /app/host-artifacts/demos/document-flow-replay
```
