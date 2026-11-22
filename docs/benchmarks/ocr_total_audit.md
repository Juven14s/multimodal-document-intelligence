# Audit de l'extraction du montant total

## Validation SROIE

Tesseract 5.3.0 a relu les 99 reçus du split de validation en anglais avec PSM 4.
Le replay v2 reproduit les résultats archivés. Les sorties Tesseract sont sauvegardées
avant le filtre de confiance, avec les coordonnées et la confiance de chaque mot.
Le replay v3 utilise les mêmes pages et les mêmes mots que v2. Il écarte des lignes
de taxe sans lien avec le montant à payer, les pourcentages et les références de carte.

| Version | Exact | Faux | Absent |
|---|---:|---:|---:|
| OCR v2 | 49 | 24 | 26 |
| OCR v3 | 51 | 19 | 29 |

Deux totaux faux deviennent exacts, trois deviennent absents et les 49 totaux exacts
sont conservés. Pour 24 des 50 échecs v2, le montant de référence apparaît dans au
moins une ligne transmise aux règles. Sa présence quelque part sur le reçu aide à
diagnostiquer la sélection. Le libellé et la position restent nécessaires pour établir
qu'il s'agit bien du total. Pour les 26 autres échecs, le montant figure dans la
transcription SROIE mais échappe à la recherche numérique, même avant le filtre de
confiance de Tesseract. Une seconde lecture des mêmes pages avec PSM 11 retrouve le
montant dans le texte de 11 de ces 26 reçus. Cet essai ciblé mesure la récupération du
texte. La sélection du total et l'effet de ce mode sur les reçus déjà bien lus restent
à mesurer.

## Contrôle externe CORD v2

Le split test officiel a été figé avant le réglage de v3. Il compte 100 reçus dont
95 avec un total de référence. Le contrôle des SHA-256 des PNG ne trouve aucun doublon
entre ces reçus ni avec le manifeste SROIE. Le protocole et la révision du jeu figurent
dans `artifacts/benchmarks/cord-total-holdout/protocol.json`. [Source CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2)

| Version | Exact | Faux | Absent |
|---|---:|---:|---:|
| OCR v2 | 21 | 6 | 68 |
| OCR v3 | 21 | 6 | 68 |

Les 95 prédictions sont identiques entre v2 et v3. Le gain observé sur SROIE ne se
retrouve pas sur CORD avec cette configuration. Dans 12 des 74 échecs, le montant
attendu apparaît déjà dans une ligne transmise aux règles. Le premier réglage v3
reste expérimental. L'API utilise v2 et son profil de calibration associé.

Les résultats détaillés se trouvent dans `artifacts/benchmarks/cord-total-holdout/`
et `artifacts/benchmarks/sroie-total-audit/`.

## Seconde lecture automatique sur les totaux manquants

L'expérience `ocr-rules-v3-psm11` utilise le même prétraitement que v3 et appelle
Tesseract en PSM 11 uniquement quand la première lecture laisse le total vide. Un
montant doit être associé à un libellé de total sur la même ligne ou à un bloc situé
à droite et aligné verticalement. Plusieurs montants distincts déclenchent une
abstention. Toute proposition de la seconde lecture reste en revue humaine.

Sur les 99 reçus SROIE, la seconde lecture se déclenche 29 fois. Elle ajoute deux
totaux exacts, ce qui porte le résultat à 53 exacts, 19 faux et 27 absents. Le montant
attendu apparaît dans le texte de 21 des 29 secondes lectures. Dans 19 de ces cas,
la règle ne trouve pas d'association suffisamment claire entre le nombre et un
libellé de total.

L'appel Tesseract supplémentaire dure en moyenne 1,06 s lorsqu'il est déclenché.
Rapporté aux 99 reçus, il ajoute 311 ms par reçu en moyenne. Ce calcul porte sur le
temps OCR des captures. Le temps de correction humaine reste à mesurer.

Sur le lot CORD déjà évalué lors de l'itération précédente, les règles de la seconde
lecture ont été figées avant le lancement. Le contrôle de transfert porte sur 95
reçus. La seconde lecture se déclenche 68 fois et propose 6 totaux en revue humaine.
Cinq sont exacts et un est faux. Le total du lot passe à 26 exacts, 7 faux et 62 absents,
avec 504 ms de temps OCR supplémentaire par reçu en moyenne.

La variante reste une expérience locale. Sur ces deux lots, la seconde lecture se
déclenche fréquemment et ne récupère qu'une partie des montants manquants. Les
résultats détaillés et les temps de chaque lecture figurent dans les deux rapports
`ocr-rules-v3-psm11.json`.

Pour rejouer cette expérience sur le cache local, ajouter `--second-pass` à la
commande v3 ci-dessous. Le résultat est séparé dans `ocr-rules-v3-psm11.json`.

## Rejouer les résultats sans Tesseract

En présence des sorties OCR archivées, la commande suivante rejoue la variante v3 sous
Windows. Pour v2, remplacer `ocr-rules-v3` par `ocr-rules-v2`.

```powershell
.\.venv\Scripts\python.exe -m document_intelligence.evaluation.ocr_total_audit `
  data\processed\sroie-full\sroie.jsonl --split validation `
  --output-dir artifacts\benchmarks\sroie-total-audit `
  --rules-version ocr-rules-v3 --replay-only
```

Les JSON de résultats contiennent le total attendu, la prédiction et les lignes OCR
pertinentes pour chaque reçu. Le dossier `ocr` conserve les mots, les lignes et les
sorties Tesseract organisés par empreinte de l'image. Pour refaire la capture depuis
la racine du projet, choisir un dossier de sortie neuf et utiliser le conteneur API.

```powershell
.\scripts\docker.ps1 compose run --rm --no-deps --pull never `
  -e PYTHONPATH=/app/src `
  -v "${PWD}/src:/app/src:ro" -v "${PWD}/data:/app/data:ro" `
  -v "${PWD}/artifacts:/app/host-artifacts" `
  api python -m document_intelligence.evaluation.ocr_total_audit `
  /app/data/processed/sroie-full/sroie.jsonl --split validation `
  --output-dir /app/host-artifacts/benchmarks/sroie-total-audit-recapture
```

## Mesurer le temps de correction

L'essai local utilise 20 reçus publics de SROIE et les propositions archivées d'OCR v2.
Chaque mode reçoit 5 reçus avec une proposition exacte, 3 avec une proposition fausse
et 2 avec une proposition absente. Les reçus sont distincts et leur ordre est tiré au
sort pour chaque session. Pour lancer l'essai depuis la racine du projet :

```powershell
.\.venv\Scripts\python.exe -m document_intelligence.ui.receipt_pilot
```

L'interface est accessible à `http://127.0.0.1:7871/`. Les observations sont enregistrées
dans `artifacts/receipt-pilot/sessions/`. Le chronomètre démarre après le chargement de
l'image. Une reprise du reçu ou un changement d'onglet exclut sa durée du calcul de la
médiane. Le bilan d'exactitude inclut tous les reçus. Ce pilote mesure un ordre de
grandeur pour le champ total, avec une proportion de cas difficiles fixée à l'avance.

### Essai de préremplissage sélectif

La politique `total-unique-label-agreement-v1` affiche un montant lorsque les lignes
OCR désignent un seul total explicite, que ce montant correspond à la proposition
OCR v2 et qu'aucune incohérence entre sous-total, taxe et total n'est détectée.
Un libellé comme `Total supplies` ne constitue pas une preuve de montant à payer.

Le filtre a été choisi sur les 99 reçus SROIE de validation. L'évaluation utilise
320 reçus du test historique, après retrait des doublons inter-splits, des reçus du
test fonctionnel et des images du premier essai humain. Les captures OCR réutilisent
les pages prétraitées d'OCR v2 et vérifient leur empreinte. Les rapports contiennent
les résultats par reçu dans `artifacts/benchmarks/sroie-selective-validation/report.json`
et `artifacts/benchmarks/sroie-selective-test/report.json`.

| Lot | Approche | Exact | Faux | Vide | Exactitude des montants affichés |
|---|---|---:|---:|---:|---:|
| Validation, 99 reçus | OCR v2 | 49 | 24 | 26 | 67,1 % |
| Validation, 99 reçus | Filtre sélectif | 15 | 0 | 84 | 100 % |
| Test, 320 reçus | OCR v2 | 162 | 102 | 56 | 61,4 % |
| Test, 320 reçus | Filtre sélectif | 50 | 11 | 259 | 82,0 % |

Le filtre affiche un montant sur 19,1 % des reçus test. Il élimine 91 fausses
propositions et masque aussi 112 propositions auparavant exactes. Les 11 erreurs
restantes sur 61 propositions justifient une vérification humaine. La différence
entre validation et test montre la fragilité de l'estimation sur 15 propositions.

Le deuxième essai tire 20 nouveaux reçus dans ce lot. Chaque groupe reçoit quatre
reçus pour lesquels le filtre fournit un montant et six où il s'abstient. Cette
proportion donne assez de propositions à vérifier pour le pilote. Le bilan ne
représente pas la fréquence naturelle de préremplissage sur l'ensemble du corpus.

```powershell
.\.venv\Scripts\python.exe -m document_intelligence.ui.receipt_pilot --study selective
```

Ouvrir `http://127.0.0.1:7872/`. Les sessions de ce pilote sont enregistrées dans
`artifacts/receipt-pilot/selective/sessions/`.

### Essai avec la fréquence de préremplissage du benchmark

Le troisième lot contient 40 reçus tirés selon la disponibilité d'une proposition,
après exclusion des images des deux premiers essais. Le filtre affiche un montant
pour quatre des 20 reçus assistés, soit 20 %, proche des 19,1 % mesurés sur le lot de
320 reçus. Les 20 reçus manuels contiennent la même proportion de cas où le filtre
aurait prérempli un montant. Pour chaque paire, l'un des reçus est affecté au mode
manuel et l'autre au mode assisté. L'ordre des paires et l'ordre des deux modes sont
tirés au sort pour chaque session. Les annotations de référence servent au score
après saisie.

```powershell
.\.venv\Scripts\python.exe -m document_intelligence.ui.receipt_pilot --study representative
```

Ouvrir `http://127.0.0.1:7873/`. Les résultats sont sauvegardés dans
`artifacts/receipt-pilot/representative/sessions/`. L'essai mesure l'effet à la
fréquence de préremplissage du benchmark SROIE, avec un échantillon de 20 reçus par
mode et un relecteur par session.

Pour un essai pilote sur des reçus représentatifs de l'usage, répartir aléatoirement
les images en deux lots comparables. Dans le premier, le relecteur saisit le total à
partir de l'image. Dans le second, il confirme ou corrige la proposition. Chaque
relecteur voit une seule fois chaque reçu. Deux relecteurs peuvent échanger les modes
entre les lots et alterner l'ordre pour limiter l'effet d'apprentissage.

Démarrer le chronomètre à l'affichage du reçu et l'arrêter à la validation du montant.
Pour chaque reçu, relever son identifiant, le mode, le temps et le montant validé.
Comparer la médiane du temps par mode et la proportion de montants encore faux ou
manquants, avec les références masquées pendant la saisie. Les mesures de Tesseract
portent sur la machine. Le gain pour le relecteur demande un essai humain.
