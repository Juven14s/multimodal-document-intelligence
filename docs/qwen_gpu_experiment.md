# Expérience Qwen3-VL sur GPU

## But

Comparer sur exactement les mêmes documents :

1. Qwen3-VL zero-shot avec le schéma JSON complet ;
2. Qwen3-VL zero-shot avec le gabarit compact ;
3. Qwen3-VL few-shot compact avec des exemples issus uniquement du train.

Le modèle est chargé une seule fois et réutilisé entre les trois passages. Cela évite de
confondre le temps de téléchargement ou de chargement avec la latence d'inférence.

## Préparation

Sur l'hôte GPU, installer le projet et vérifier CUDA :

```powershell
python -m pip install -e ".[vlm]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Le script PowerShell utilise par défaut le manifeste complet et son catalogue few-shot.
Si ce catalogue n'existe pas encore :

```powershell
docintel-build-fewshot data\processed\sroie-full\sroie.jsonl `
  --data-root data\processed\sroie-full `
  --output data\processed\sroie-full\fewshot.jsonl
```

## Exécution

Sous Windows PowerShell :

```powershell
.\scripts\qwen-gpu-experiment.ps1 `
  -Manifest data\processed\sroie-full\sroie.jsonl `
  -Examples data\processed\sroie-full\fewshot.jsonl `
  -ModelId Qwen/Qwen3-VL-2B-Instruct `
  -Limit 10 -FewShotCount 2 -Overwrite
```

Sous Linux ou dans un notebook GPU :

```bash
python -m document_intelligence.evaluation.qwen_experiment \
  data/processed/sroie-full/sroie.jsonl \
  --examples data/processed/sroie-full/fewshot.jsonl \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --limit 10 --few-shot-count 2 --overwrite
```

Sans CUDA, la commande s'arrête avant le chargement des poids. `--allow-cpu` existe pour
un smoke test volontaire, mais ne doit pas servir à comparer les latences de production.

## Sorties

`artifacts/qwen-experiment/experiment.json` regroupe :

- versions de Python, PyTorch et Transformers ;
- disponibilité et version CUDA ;
- nom et mémoire de chaque GPU ;
- modèle, manifeste, taille d'échantillon et dates du passage ;
- rapport complet de chaque stratégie.

Chaque stratégie possède aussi son propre dossier :

```text
artifacts/qwen-experiment/
├── zero-shot-schema/
├── zero-shot-compact/
├── few-shot-compact/
└── experiment.json
```

Chaque job conserve `vlm-trace.json`, y compris lorsque la réponse n'est pas conforme au
contrat. Cette trace contient la réponse brute, l'empreinte du prompt, sa version, la
validité syntaxique du JSON, la conformité Pydantic, la latence et la mémoire GPU maximale.

## Reproductibilité et absence de fuite

- le split évalué est `test` ;
- le catalogue few-shot ne contient que `train` ;
- le sélecteur exclut l'identifiant cible et toute image de même SHA-256 ;
- `--overwrite` recommence les trois passages ; sans cette option, les prédictions déjà
  terminées sont reprises ;
- les réponses invalides restent dans les traces et sont comptées comme échecs, jamais
  supprimées du dénominateur.
