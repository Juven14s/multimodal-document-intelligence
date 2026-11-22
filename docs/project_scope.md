# Périmètre et critères de succès

## Décision de cadrage

Le premier benchmark porte sur les **reçus**, intersection exploitable de CORD et SROIE.
Les factures complexes constituent une extension : les ajouter dès le départ mélangerait
des taxonomies et des niveaux d'annotation différents, ce qui rendrait la comparaison
OCR/VLM difficile à interpréter.

## Unité de traitement

- Entrée : une image ou un PDF représentant un document.
- Sortie : un objet `DocumentExtraction` valide, même en cas d'échec partiel.
- Traçabilité : chaque valeur conserve le texte brut, une confiance et ses preuves.
- Abstention : le système peut retourner `null` et demander une revue humaine.

## Champs MVP

| Groupe | Champs |
|---|---|
| Identité | type de document, commerçant, numéro de document |
| Dates | date d'émission, date d'échéance |
| Montants | devise, sous-total, taxe, total |
| Détail | description, quantité, prix unitaire, montant de ligne |

## Garde-fous

- `subtotal + tax ~= total`, avec tolérance configurable.
- somme des lignes cohérente avec le sous-total lorsqu'elle est disponible.
- confiance et preuves évaluées par champ, pas seulement au niveau document.
- toute violation bloquante ou confiance inférieure au seuil déclenche une revue.

## Mesures

- Exact match normalisé et F1 par champ.
- Taux de JSON valide et taux de champs absents/hallucinés.
- Taux de documents envoyés en revue et précision sur les documents auto-acceptés.
- Latence p50/p95, mémoire GPU maximale et coût moyen par document.

## Définition de terminé du MVP

1. Le même jeu de test figé compare OCR, VLM zero-shot et VLM few-shot.
2. Les résultats sont reproductibles depuis une commande et tracés dans MLflow.
3. Un seuil de revue est choisi sur validation, jamais sur test.
4. L'API et l'interface montrent les preuves et permettent une correction humaine.
5. Les corrections sont historisées sans écraser la prédiction d'origine.
