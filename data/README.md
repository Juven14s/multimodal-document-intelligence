# Données

Les données brutes restent locales et ne sont pas versionnées.

Disposition prévue :

```text
data/
  raw/          # téléchargements CORD et SROIE inchangés
  interim/      # images rendues, OCR et annotations harmonisées
  processed/    # manifestes train/validation/test
```

## Sources et couverture

| Dataset | Splits source | Vérité terrain principale |
|---|---|---|
| CORD v2 | 800 train, 100 validation, 100 test | lignes, sous-total, taxe, total |
| SROIE | 626 train source, 361 test dans le snapshot préparé | commerçant, adresse, date, total |

Les champs absents de la taxonomie source ne sont pas considérés comme des erreurs. Chaque
annotation contient donc `annotated_fields`, le masque utilisé lors de l'évaluation.

## Préparation

```powershell
python -m pip install -e ".[data,dev]"
docintel-prepare cord_v2 --limit-per-split 10
docintel-prepare sroie --limit-per-split 10
```

Sans `--limit-per-split`, l'ensemble du dataset est traité. La commande crée :

- un JSONL canonique dans `data/processed` ;
- un rapport d'audit JSON ;
- les images PNG sous `data/processed/images` ;
- un SHA-256 par image dans le manifeste.

Les tests officiels restent intacts. Pour SROIE, 15 % du split d'entraînement est affecté
de manière déterministe à la validation. CORD conserve son split de validation officiel.
Le rapport d'audit regroupe les images identiques par SHA-256 et indique les identifiants
présents dans plusieurs splits. Avant une évaluation indépendante, réserver des images
distinctes de celles utilisées pour ajuster les règles, entraîner et calibrer le modèle.
