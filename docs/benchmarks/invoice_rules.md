# Extraction des factures

## Protocole

Cinq factures synthétiques et leurs références ont été figées avant l'ajout du profil
`ocr-invoice-v1`. Les six pages ont été rendues avec Poppler et inspectées visuellement.
Les mêmes PDFs ont été envoyés à l'API FastAPI avec OCR v2 puis `ocr-invoice-v1` dans
Docker. Les scores ci-dessous portent sur les résultats avant correction humaine.
[Factures et empreintes](../../output/pdf/invoice-holdout/protocol.json)

| Vérification | OCR v2 | OCR factures v1 |
|---|---:|---:|
| Total facturé ou abstention attendue | 4/5 | 5/5 |
| Solde restant ou abstention attendue | 2/5 | 5/5 |
| Devise ou abstention attendue | 3/5 | 5/5 |
| Date de facture | 4/5 | 5/5 |
| Date d'échéance | 0/5 | 5/5 |
| Vérifications conformes | 13/25 | 25/25 |

Le profil distingue 300,00 EUR facturés de 200,00 EUR restant à payer dans la facture
de démonstration avec acompte. Il garde une ligne OCR distincte comme preuve pour
chaque montant. Il laisse `total.value` vide quand seul le solde est imprimé et
laisse `currency` vide lorsque deux devises déclarées se contredisent.

Le résultat porte sur cinq factures synthétiques. Il reste trois écarts sur les autres
champs du lot, concernant deux numéros de facture et un sous-total. Le profil garde
les factures en revue humaine. Le cache OCR, les documents clients et la base de revue
restent locaux.

## Rejouer avec Docker

Cette commande génère un rapport local avant revue. Omettre `--invoice-rules` et choisir
un autre dossier de sortie pour mesurer la référence OCR v2.

```powershell
.\scripts\docker.ps1 compose run --rm --no-deps --pull never `
  -e PYTHONPATH=/app/src -v "${PWD}/src:/app/src:ro" `
  -v "${PWD}/scripts:/app/demo-scripts:ro" `
  -v "${PWD}/output/pdf/invoice-holdout:/app/fixtures:ro" `
  -v "${PWD}/artifacts:/app/host-artifacts" `
  api python /app/demo-scripts/document_flow_demo.py `
  --invoices /app/fixtures --expected-protocol /app/fixtures/protocol.json `
  --extract-only --invoice-rules --output /app/host-artifacts/benchmarks/invoice-replay
```
