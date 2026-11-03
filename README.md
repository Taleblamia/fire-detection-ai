# D?tection de feu par intelligence artificielle

Projet universitaire de vision par ordinateur : classification d'images en trois cat?gories, **feu**, **absence de feu** et **d?but d'incendie**, avec traitement d'une vid?o et visualisation Grad-CAM.

## Mod?le et fonctionnement

Le notebook d'entra?nement adapte un r?seau **MobileNetV2 pr?entra?n?** avec PyTorch et PyTorch Lightning. Les images sont redimensionn?es ? 224 ? 224 pixels. La configuration enregistr?e utilise DB3, un partage entra?nement/validation de 80/20, des lots de 64 et jusqu'? 50 ?poques avec arr?t anticip?. Le mod?le est export? au format TorchScript (`model_jit.pth`).

Le traitement vid?o utilise OpenCV et ignore les images trop similaires selon le SSIM. Il attribue une classe ? chaque image retenue ; il ne produit pas de bo?tes de localisation du feu.

## Contenu

- `notebooks/2_Fire_classification_torchlightning_Input_2024.ipynb` : entra?nement, ?valuation, export et analyse vid?o.
- `notebooks/6_Model_inference_Kaggle (1).ipynb` : pr?dictions sur les images du challenge et g?n?ration du CSV.
- Notebook `Explainable_Fire_Detection` : cartes Grad-CAM avec Captum.
- `results/` : deux fichiers de soumission, contenant chacun 1 541 pr?dictions (`ID`, `Label`).

## R?sultats archiv?s

| Exp?rience | Score indiqu? dans les archives locales |
| --- | --- |
| MobileNetV2 | 0,884 |
| EfficientNet-B0 | 0,839 |

Ces valeurs proviennent des noms des dossiers d'exp?riences et ne sont pas v?rifi?es ici sur Kaggle. La m?trique exacte n'est pas document?e dans les fichiers CSV ; ceux-ci ne contiennent pas les r?ponses de r?f?rence. Le dossier MobileNetV2 mentionne ?galement une 5e place, ? confirmer sur le classement d'origine.

## Utilisation

Les notebooks ont ?t? con?us pour **Google Colab** : ils contiennent des commandes Linux, des t?l?chargements de jeux de donn?es, une connexion W&B et des appels ? `google.colab`.

1. Ouvrir le notebook d'entra?nement dans Colab et s?lectionner un GPU si disponible.
2. Ex?cuter les cellules d'installation, de pr?paration des donn?es et d'entra?nement. Les liens de t?l?chargement doivent ?tre encore accessibles et l'utilisation des donn?es doit respecter leurs conditions.
3. R?cup?rer `model_jit.pth`, puis le charger dans le notebook d'inf?rence pour g?n?rer `submission.csv`.
4. Pour la vid?o, fournir `video.mp4` et adapter son chemin.

`requirements.txt` liste les d?pendances Python utilis?es, sans figer les versions. Il ne remplace pas les adaptations n?cessaires pour ex?cuter les notebooks hors de Colab.

### Explicabilit? : adaptation n?cessaire

Le notebook Grad-CAM d?finit actuellement **EfficientNet-V2-S** et attend `testmodel-sidi.ckpt`, absent du d?p?t. Pour expliquer le MobileNetV2 entra?n?, il faut adapter cette architecture et charger un checkpoint compatible. Ce notebook n'est donc pas directement ex?cutable avec le seul export TorchScript.

## Fichiers exclus

Les poids de mod?les (`.pth`, `.ckpt`), jeux de donn?es, vid?o et supports p?dagogiques ne sont pas inclus. Les sorties d'ex?cution des notebooks publi?s ont ?t? retir?es. Les fichiers originaux restent dans le dossier de travail local.

## Auteure

Lamia Taleb ? projet r?alis? dans le cadre du D?fi 1, ? partir des supports p?dagogiques du Pr. Sidi Mahmoudi (UMONS).
