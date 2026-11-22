# Dataset LoRA/PEFT multimodal

## État actuel

Le corpus SROIE complet est exporté dans un format conversationnel compatible avec un
entraînement supervisé multimodal :

- 527 exemples d'entraînement ;
- 99 exemples de validation ;
- 0 exemple de test ;
- chemins d'images relatifs pour permettre de déplacer le projet sur une machine GPU.

Les fichiers se trouvent dans `data/processed/sroie-full/sft/`. Le rapport
`dataset-report.json` contient le nombre d'exemples et le SHA-256 de chaque JSONL.

## Reproduire l'export

```powershell
.\scripts\export-sft.ps1
```

Chaque ligne contient les messages utilisateur (image et consigne) puis la réponse
assistant conforme au contrat `VLMResponse`. L'exporteur n'accepte que les splits
`train` et `validation` et lève une erreur si un enregistrement `test` lui est fourni.

## Entraînement

Le fine-tuning LoRA n'est pas lancé sur cette machine : PyTorch y est installé en mode
CPU et aucun périphérique CUDA n'est disponible. Les artefacts sont prêts à être copiés
sur une machine GPU. Avant un entraînement, conserver la validation pour le choix des
hyperparamètres et ne mesurer le test qu'une fois la configuration figée.
