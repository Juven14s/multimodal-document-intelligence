# Smoke test VLM CPU — prompting et few-shot

Date d'exécution : 21 novembre 2026

## Objectif

Valider une inférence multimodale réelle sur la machine locale avant d'engager le
téléchargement et l'exécution de Qwen3-VL-2B. Ce test ne remplace pas le benchmark Qwen :
il vérifie le chemin image → modèle → réponse brute → JSON → contrat Pydantic.

## Environnement

- modèle : `HuggingFaceTB/SmolVLM-256M-Instruct` ;
- document : `sroie-X51005230659`, un reçu du split test SROIE ;
- périphérique : CPU, PyTorch sans CUDA ;
- mémoire physique : 15,8 Go, dont 4,1 Go disponibles lors du précontrôle ;
- budget : 512 nouveaux tokens ;
- coût d'API : 0 USD.

Le modèle Qwen3-VL-2B cible représente environ 4,27 Go de poids natifs. Son benchmark
est différé vers un environnement GPU afin d'éviter une exécution locale instable et non
représentative.

## Résultats

| Stratégie | Latence | JSON syntaxique | Schéma Pydantic | Observation |
|---|---:|---:|---:|---|
| Schéma complet zero-shot | 55,25 s | Non | Non | Réponse OCR en texte libre |
| Gabarit compact zero-shot | 52,38 s | Oui | Non | JSON avec schéma propre au modèle |
| Gabarit compact one-shot | 87,51 s | Oui | Non | Même schéma alternatif malgré l'exemple |

Le gabarit compact améliore donc la validité syntaxique, mais ce VLM de 256M paramètres
n'obéit pas au contrat métier demandé. Le few-shot augmente la latence de 67 % par rapport
au compact zero-shot sans rendre la sortie conforme.

Le modèle lit néanmoins correctement plusieurs éléments visibles, notamment le nom
`SWC Enterprise SDN BHD`, une partie de l'adresse et le total `8.00`. L'échec porte sur le
respect de la structure, pas sur l'absence totale de perception du document.

## Traçabilité

Les réponses brutes, empreintes des prompts et erreurs de validation se trouvent sous :

- `artifacts/vlm-smoke/smolvlm-sroie-X51005230659/` ;
- `artifacts/vlm-smoke/smolvlm-compact-sroie-X51005230659/` ;
- `artifacts/vlm-smoke/smolvlm-fewshot-sroie-X51005230659/`.

L'exemple one-shot `sroie-X00016469676` provient du split d'entraînement. Le document
cible et les images de même empreinte sont exclus par le sélecteur afin d'éviter toute
fuite de données.

## Décision

SmolVLM-256M reste un test fonctionnel CPU et n'est pas retenu comme modèle candidat.
La comparaison qualité suivante doit exécuter Qwen3-VL-2B-Instruct sur les mêmes dix
documents avec, dans l'ordre : schéma complet zero-shot, gabarit compact zero-shot, puis
few-shot issu exclusivement du train.
