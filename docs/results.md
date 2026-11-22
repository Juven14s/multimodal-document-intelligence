# Bilan du prototype CPU

Le MVP couvre le dépôt PDF/image, OCR, extraction structurée, revue dans Gradio,
historisation SQLite et export JSON. L'API, l'interface et MLflow démarrent en local
avec Docker. Les documents synthétiques permettent d'essayer le parcours depuis un
dépôt vierge.

| Expérience | Résultat | Périmètre |
|---|---|---|
| OCR v2 sur les reçus SROIE | 50,6 % de totaux exacts et 10,3 % de reçus entièrement exacts | 340 reçus après retrait des images déjà vues et des doublons, recalcul rétrospectif |
| Préremplissage sélectif | 50 totaux exacts sur 61 affichés, sur 320 reçus | 82 % d'exactitude parmi les propositions et 19 % de couverture |
| Saisie assistée à cette fréquence | 18 totaux corrects sur 20 dans chaque mode, 3,4 s gagnées au total | Un relecteur et 20 reçus par mode, résultat indicatif |
| Profil OCR factures | 25 vérifications prioritaires sur 25, contre 13 pour OCR v2 | 5 factures synthétiques figées, abstentions attendues incluses |
| SmolVLM CPU | 0 extraction conforme sur 10 en zero-shot, un essai few-shot invalide | Le lot few-shot a été interrompu après le premier résultat complet |

L'exactitude de l'extraction et le temps passé en revue sont deux mesures distinctes.
Le dernier essai humain ne démontre pas un gain de temps substantiel. Le score factures
mesure le fonctionnement sur des scénarios synthétiques. Une mesure sur des documents
représentatifs de l'usage est nécessaire avant un déploiement métier.

Les essais locaux OCR et SmolVLM ont tourné sans API d'inférence payante. La consommation
électrique et le coût du temps humain restent à mesurer. Qwen3-VL et LoRA dépendent
d'une machine CUDA. Le protocole GPU et l'export SFT sont disponibles. Aucun résultat
GPU ou LoRA n'est revendiqué.

- [Audit SROIE et test sélectif](benchmarks/ocr_total_audit.md)
- [Recalcul sur les 340 reçus](benchmarks/sroie_ocr_iteration_3.md)
- [Démo produit](benchmarks/document_flow_demo.md)
- [Profil factures](benchmarks/invoice_rules.md)
- [Comparaison SmolVLM CPU](benchmarks/smolvlm_cpu_10.md)
- [Expérience Qwen sur CUDA](qwen_gpu_experiment.md)
- [Export du corpus LoRA](lora_dataset.md)
