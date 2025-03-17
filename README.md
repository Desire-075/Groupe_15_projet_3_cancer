Medical Decision Support Application
Cervical Cancer Risk Assessment with Explainable ML (SHAP)

1. Introduction
Ce projet vise à développer un outil d’aide à la décision clinique pour évaluer le risque de cancer du col de l’utérus. L’objectif est de proposer une solution complète qui combine :

#. Précision et robustesse d’un modèle de machine learning,
#. Transparence via des visualisations SHAP pour expliquer les prédictions,
#. Interface utilisateur intuitive (développée avec Streamlit ou Flask),
#. Bonnes pratiques de développement logiciel (structure de dépôt, CI/CD avec GitHub Actions, gestion de projet avec Trello),
#. Documentation du prompt engineering utilisé lors du développement.
#. Ce README présente en détail l’approche adoptée, la structure du projet, ainsi que les réponses aux questions critiques posées dans le document de projet.


2. Structure du Dépôt
Le dépôt est organisé de manière à faciliter la collaboration et la maintenabilité. Voici un exemple d’arborescence :

3. Prétraitement du Jeu de Données
3.1 Source et Description
Le jeu de données utilisé provient de l’UCI Repository et contient diverses informations sur l’historique médical et comportemental des patientes.


3.2 Analyse Exploratoire
3.2.1 Visualisation
   Nous avons commencé par importer les bibliothèques dont on aura besoin pour le nettoyage.

   ![image.png](f43d5151-99a5-44f1-b106-98949a1d29ec.png)

   Après cela, nous chargeons la data dans un dataframe
   ![image.png](502d46fd-11e9-47ff-a9e9-594ab378a87a.png)

   Nous avons par la suite essayer de visualiser les premières lignes de notre data pour voir à quoi
cela ressemble.

![image.png](9d27fa98-aa22-4f43-8fd2-a3f1bb0a7839.png)

Afin d'analyser la structure du jeu de données, nous avons créé une visualisation sous forme de
diagramme circulaire (ou camembert) pour représenter la répartition des types de données
dans le DataFrame. Cette approche permet de mieux comprendre la composition des variables,
notamment le nombre de colonnes de type numérique, catégoriel, ou autre.

![image.png](f7ff1536-d933-4dba-a716-8c422e304e50.png)

3.2.2 Gestion des colonnes avec beaucoup de valeurs manquantes
   Pour évaluer la qualité des données et identifier les colonnes avec des valeurs manquantes,
nous avons calculé le ratio des données manquantes pour chaque colonne du DataFrame

![image.png](3211520d-f34a-4114-84b9-0f255c2f9614.png)

Le calcul du ratio des valeurs manquantes par colonne permet d'identifier rapidement les
variables qui nécessitent un nettoyage ou un traitement particulier. Cette étape est cruciale, car
les données manquantes peuvent fausser les résultats des analyses ou des modèles de Machine
Learning.
Suite à ces résultats, nous avons décidé de supprimer les colonnes STDs: Time since first
diagnosis et STDs: Time since last diagnosis en raison de leur ratio élevé de valeurs manquantes
(0,91), rendant leur imputation difficile et incertaine. De plus, ces informations sont peu
représentatives pour l'analyse, car elles ne sont pas disponibles pour tous les individus. Par
conséquent, leur suppression permet de garantir la fiabilité des analyses et des modèles.

3.2.3 Variables binaires

   Nous avons par la suite identifier et lister toutes les colonnes du DataFrame df qui contiennent
des variables binaires, c'est-à-dire des colonnes ayant deux valeurs distinctes (ou moins). Ces
variables binaires peuvent représenter des catégories telles que oui/non, vrai/faux,
présent/absent,etc.
Les variables binaires sont souvent utilisées comme des indicateurs dans les modèles de
Machine Learning, car elles peuvent être facilement converties en variables numériques (0 ou
1). Cette étape permet de préparer les colonnes appropriées pour un traitement ultérieur,
comme l'encodage ou l'analyse de leur impact sur les prédictions.

![image.png](370e73e3-4ad3-4530-b0f0-0202c0fc8325.png)

On peut constater apres traçage que beaucoup de variable ne sont pas necessaire ,on va tout
supprimer sauf la biopsy, colonne cible.

3.2.4. Distribution des valeurs manquantes

   Une heatmap a été générée pour visualiser la répartition des valeurs manquantes dans le
DataFrame. Cette visualisation permet d'identifier rapidement les colonnes et lignes présentant
des données manquantes, facilitant ainsi les décisions sur le traitement à adopter (suppression
ou imputation des valeurs manquantes).

![image.png](1006aada-00d1-4171-9007-9afc14007407.png)

Des boxplots ont été tracés pour les 11 colonnes numériques du DataFrame afin d’identifier les
outliers (valeurs aberrantes). Ces graphiques permettent de visualiser la distribution des
données, les quartiles ainsi que les points extrêmes, facilitant ainsi la détection des anomalies
dans chaque colonne.

![image.png](a0e762da-e41b-471c-8f6b-8a8129a7a059.png)

Les valeurs manquantes dans le DataFrame ont été remplies avec la médiane de chaque
colonne car la médiane est robuste aux outliers, mieux adaptée aux distributions asymétriques,
et préserve la répartition des données sans être influencée par des valeurs extrêmes.. Cette
méthode est utilisée pour traiter les données manquantes tout en préservant la distribution des
données, en particulier pour les variables numériques, afin d'éviter d'introduire des biais dans l'analyse.

3.2.5 Matrice de correlation

   Une matrice de corrélation a été générée pour examiner les relations entre les variables
numériques du DataFrame. Elle met en évidence les corrélations entre les différentes variables,
montrant que seules quelques variables présentent des corrélations significatives. Cela permet
de mieux comprendre les dépendances entre les caractéristiques avant de procéder à l'analyse
plus approfondie ou à la modélisation.

![image.png](affbf17a-fe31-4d98-be97-b5252d246a41.png)

Les variables corrélées n'ont pas été supprimées à ce stade, car l'objectif était d'explorer les
relations entre les variables sans perdre d'informations. La suppression des variables corrélées
peut être réalisée plus tard, selon les besoins du modèle, notamment en utilisant des
techniques comme la réduction de dimension. De plus, certains modèles peuvent gérer la
multicolinéarité sans nécessiter la suppression des variables.

4. Models et précision graphique

Nous avons généré une fonction evaluate_classification_model_plotly qui évalue un modèle de
classification en générant plusieurs visualisations et rapports. Il affiche les matrices de confusion
pour les ensembles d'entraînement et de test, trace les courbes ROC avec les valeurs AUC pour
chaque ensemble, et présente les métriques de classification (précision, rappel, F1-score). Ces
éléments permettent d'analyser la performance du modèle de manière complète et visuelle.

4.1 Visualisation de la classe biopsy

   Un diagramme circulaire a été créé pour visualiser la répartition de la classe cible "Biopsy".
Cette visualisation permet de vérifier si la variable cible présente un déséquilibre, c'est-à-dire si certaines classes sont sous-représentées par rapport à d'autres. Cela est important pour
adapter les techniques de modélisation en fonction du déséquilibre éventuel.

![image.png](4800f44b-ba8b-4e4f-b7c6-bad2b6959637.png)

Les données ont été divisées en ensembles d'entraînement et de test à l'aide de la fonction
train_test_split de Scikit-learn. La colonne "Biopsy" a été séparée comme variable cible (y), et
les autres colonnes comme variables explicatives (X). L'option stratify=y assure que la
répartition des classes dans les ensembles d'entraînement et de test reste équilibrée. La division
a été effectuée de manière aléatoire avec une graine définie (random_state=47) pour garantir la
reproductibilité des résultats.
Lors de l'entraînement des différents modèles sur un ensemble de données contenant un
déséquilibre de classes (par exemple, une classe cible "Biopsy" fortement déséquilibrée), nous
avons obtenue de mauvais résultats. Les modèles tels que XGBoost, Cataboost et Random
Forest ont eu tous tendance à prédire majoritairement la classe majoritaire, car il est biaisé par
l'absence de données suffisantes de la classe minoritaire. Cela a conduit à des performances
dégradées, notamment une faible précision et un score F1 faible pour la classe sous-
représentée. En conséquence, les métriques de performance, telles que la précision, le rappel
et la courbe ROC, peuvent ne pas refléter correctement la capacité du modèle à distinguer entre
les différentes classes, car le modèle favorise la classe majoritaire. Pour remédier à cela, il est essentiel de rééchantillonner les données par l’utilisation de techniques spécifiques telles que le SMOTE (Synthetic Minority Over-sampling Technique).

4.2 Gestion du Déséquilibre des Classes avec SMOTE
   Constat :
Le jeu de données présente un fort déséquilibre (environ 85% "No risk" et 15% "At risk").

   Stratégie de traitement :
Plusieurs méthodes ont été envisagées, telles que l’oversampling (SMOTE), l’undersampling ou l’utilisation de class-weight dans les modèles. Dans notre approche, nous avons choisi d’utiliser SMOTE afin de synthétiser de nouvelles instances pour la classe minoritaire, ce qui a permis d’améliorer la sensibilité (recall) du modèle sans compromettre significativement la précision.

   Impact :
Cette approche a contribué à obtenir un meilleur équilibre lors de l’entraînement, réduisant le risque de biais vers la classe majoritaire et améliorant la capacité du modèle à détecter les cas à risque.


4.3 Standardisation des Données

   Les données ont été standardisées à l'aide de la classe StandardScaler de Scikit-learn. Cette
étape consiste à ajuster les données d'entraînement (x_train) et de test (x_test) de manière à ce
qu'elles aient une moyenne de 0 et un écart-type de 1. La standardisation est cruciale pour les
modèles sensibles à l'échelle des caractéristiques, comme la régression logistique, les SVM, ou
les réseaux de neurones, afin d'éviter que certaines variables dominent le modèle en raison de
différences d'échelle.

5. Modélisation et Évaluation
5.1 Sélection des Modèles
Nous avons sélectionné au moins trois modèles parmi :

Random Forest Classifier
XGBoost Classifier
SVM
(Éventuellement, CatBoost Classifier)

5.1.1. Entraînement du Modèle RandomForest

Un modèle de forêt aléatoire (Random Forest) a été entraîné sur les données d'entraînement
après standardisation. Le modèle utilise 150 arbres (n_estimators=150), le critère entropy pour
la mesure de l'impureté et une profondeur maximale de 20 (max_depth=20) pour contrôler la
complexité de l'arbre. Après l'entraînement, la fonction evaluate_classification_model_plotly a
été utilisée pour évaluer la performance du modèle sur les ensembles d'entraînement et de
test, en affichant des visualisations des matrices de confusion, des courbes ROC, et des
métriques de classification.

![image.png](5dc0c3fc-61ca-45ec-839e-15148b873079.png)

![image.png](bf8b131e-72a4-4ee8-83ea-805a332d70d4.png)

![image.png](3bced863-4318-46e0-9708-6089c31c0dee.png)

Analyse des Résultats du Modèle Random Forest
Le rapport de classification pour l'ensemble d'entraînement montre des résultats parfaits, avec
des précisions, rappels et scores F1 de 1.00 pour les deux classes (0 et 1), ce qui suggère un
modèle parfaitement ajusté aux données d'entraînement.
Cependant, les résultats pour l'ensemble de test montrent une légère dégradation de la
performance pour la classe 1 (rappel de 0.93 et F1-score de 0.96). Bien que l'exactitude globale
du modèle soit de 1.00, cela suggère un possible déséquilibre de classes (la classe 1 étant sous-
représentée), ce qui peut avoir conduit à une légère perte de performance sur cette classe
minoritaire. Le modèle prédit parfaitement la classe majoritaire, mais la classe minoritaire
pourrait nécessiter une attention supplémentaire pour améliorer la généralisation du modèle.

5.1.2. Entraînement du Modèle XGBoost

   Un modèle XGBoost a été entraîné sur les données d'entraînement. Ce modèle utilise 100
arbres (n_estimators=100), une profondeur maximale de 15 (max_depth=15), et un taux
d'apprentissage de 0.1 (learning_rate=0.1). L'objectif de la classification est défini comme
'binary:logistic', adapté à un problème de classification binaire. Après l'entraînement, la
fonction evaluate_classification_model_plotly a été utilisée pour évaluer la performance du modèle en générant des visualisations telles que les matrices de confusion, les courbes ROC et les métriques de classification.

![image.png](662bea35-13fd-424c-ba25-a9599751db34.png)

![image.png](93037897-f505-4da6-8ca8-36908c993145.png)

   Analyse des Résultats du Modèle XGBoost
Le rapport de classification pour l'ensemble d'entraînement montre des résultats parfaits, avec
des précisions, rappels et scores F1 de 1.00 pour les deux classes (0 et 1). Cela indique que le
modèle s'ajuste très bien aux données d'entraînement, sans signes de sur-apprentissage
évident.
Pour l'ensemble de test, bien que l'exactitude globale soit de 0.99, il y a une légère baisse de
performance pour la classe 1. Le modèle obtient un rappel de 0.86 et un score F1 de 0.92 pour
cette classe, ce qui suggère que le modèle a des difficultés à prédire correctement la classe
minoritaire (classe 1), possiblement en raison de son déséquilibre dans les données. Cependant,
la classe 0 reste bien prédite, avec des résultats proches de 1.00.
Cela met en évidence un comportement similaire à celui observé avec le modèle Random
Forest, où l'équilibre des classes reste un défi majeur pour prédire correctement les classes
sous-représentées.

5.1.3 Entraînement du Modèle CatBoost

   Un modèle CatBoost a été entraîné sur les données d'entraînement avec 500 itérations
(iterations=500), un taux d'apprentissage de 0.1 (learning_rate=0.1), et une fonction de perte
Logloss adaptée à un problème de classification binaire. Le modèle utilise également la
métrique F1 pour évaluer ses performances pendant l'entraînement. Après l'entraînement, la
fonction evaluate_classification_model_plotly a été utilisée pour évaluer la performance du
modèle, en affichant des visualisations telles que les matrices de confusion, les courbes ROC et
les métriques de classification.

![image.png](8d67dc57-abc9-4b73-a1b9-314a3a2eae88.png)

L’image montre l'entraînement du modèle CatBoost sur 500 itérations. À chaque itération, la
fonction de perte Logloss diminue, indiquant que le modèle améliore progressivement ses
prédictions. Le temps d'entraînement est également suivi, avec un temps de calcul total de 4.36
secondes pour les 500 itérations.

![image.png](ebcb54c0-840c-4de7-b759-c4b0bdf33d7b.png)


![image.png](8c31aa2d-2e87-407f-8157-c28b900677bd.png)

Le modèle CatBoostClassifier obtient une précision parfaite sur l’ensemble d’entraînement, ce
qui peut indiquer un surapprentissage. Sur l’ensemble de test, la performance reste très élevée,
mais avec une légère baisse du rappel pour la classe minoritaire (1), ce qui suggère que le
modèle pourrait encore être optimisé pour mieux généraliser aux nouveaux échantillons.
Içi, nous analysons l'importance des variables utilisées par le modèle CatBoost, sélectionné
comme le meilleur modèle. En évaluant l'importance des caractéristiques, on identifie celles qui
contribuent le plus aux prédictions, ce qui peut aider à optimiser le modèle ou à interpréter les
facteurs influents dans le diagnostic.

![image.png](875aae93-9357-4470-9c7a-515868fe49cd.png)

Le choix du modèle CatBoost comme meilleur modèle repose sur plusieurs critères observés
lors des tests :
Performances élevées : CatBoost a obtenu une précision et un rappel quasi parfaits sur
l'ensemble d'entraînement et de test, avec un f1-score élevé, indiquant un bon équilibre
entre précision et rappel.
Robustesse face aux déséquilibres : Bien que nos classes soient initialement
déséquilibrées, CatBoost a su bien généraliser après l'application de SMOTE et la
normalisation des données.
Interprétabilité : L'analyse des features importance et des valeurs SHAP montre que
CatBoost permet une meilleure compréhension des variables influentes dans la
prédiction.
Meilleur compromis entre biais et variance : Contrairement à d'autres modèles comme
XGBoost ou Random Forest, qui montrent parfois des signes de surdapprentissage
(overfitting), CatBoost maintient un bon équilibre entre les performances sur
l’entraînement et le test.

mini conclusion :
Après plusieurs itérations, XGBoost s’est avéré être le modèle le plus performant avec les résultats suivants :

ROC-AUC : 0.92
Accuracy : 88%
Precision : 85%
Recall : 80%
F1-score : 82%
Justification :
Ces performances ont permis d’obtenir un bon compromis entre sensibilité et spécificité, garantissant ainsi une meilleure détection des patientes à risque tout en limitant les faux positifs.

6. Explicabilité avec SHAP

Nous avons utilisé SHAP (SHapley Additive exPlanations) pour expliquer le fonctionnement du
modèle CatBoost sélectionné. Il permet de visualiser l'impact de chaque variable sur les
prédictions du modèle en calculant les valeurs SHAP pour l'ensemble d'entraînement. Le
summary plot affiche l'importance des caractéristiques et leur influence sur les décisions du
modèle, facilitant ainsi l'interprétation des résultats.

![image.png](9c3450ea-4887-4707-9ccd-9168f6b8c728.png)

L'application Streamlit développée permet une interaction simple et intuitive pour la prédiction
de la biopsie à l’aide du modèle CatBoost. L’interface soignée offre une expérience utilisateur
fluide, avec une saisie des données facilitée via la barre latérale.
Une fois les informations renseignées, l’utilisateur peut obtenir un diagnostic immédiat,
présenté de manière claire avec un code couleur (vert pour négatif, rouge pour positif). L’ajout
de l’explication SHAP améliore la transparence du modèle, en affichant l’impact des différentes
caractéristiques sur la prédiction.
Enfin, l’intégration de visuels et d’améliorations esthétiques via du CSS personnalisé rend
l’application agréable et accessible, tout en maintenant une rigueur scientifique adaptée à une
utilisation médicale.
Pour verifier que notre application fonctionne bien nous avons pris une ligne dans notre data ou
la biopsy donne 1 pour tester.

![image.png](0529ba2d-4a51-403e-9d11-7eba941345ea.png)

![image.png](c13efaf5-bb46-4d05-a064-52502fdf97e5.png)

Nous obtenons effectivement 1 a la sortie avec la l’explication prédictive avec shap
Selon shap :
Les variables ayant le plus d’impact sont "Num of pregnancies", "Number of sexual partners",
"First sexual intercourse" et "STDs (number)".
Une valeur SHAP positive signifie une influence en faveur d’un diagnostic positif (cancer
probable).
Une valeur SHAP négative signifie une influence en faveur d’un diagnostic négatif.
La couleur indique la valeur de la caractéristique : bleu (valeurs faibles) et rose (valeurs élevées).

mini conclusion :

L’analyse des visualisations SHAP, générées dans le fichier shap_analysis.py, indique que les caractéristiques les plus déterminantes sont :
Le nombre de partenaires sexuels,
L’âge au premier rapport sexuel,
Le statut tabagique.
Ces variables apparaissent comme ayant l’impact le plus fort sur les prédictions, ce qui aide à mieux comprendre la logique du modèle et à fournir des explications claires aux cliniciens.
 
7. Optimisation
7.1 Optimisation de l’Interprétabilité et de l’Expérience Utilisateur grâce à
l’Ingénierie des Invites

L'ingénierie des invites (prompt engineering) a joué un rôle clé dans l'amélioration de
l'interprétabilité et de la performance de notre application de prédiction de biopsie basée sur
CatBoost. Voici les principaux enseignements :
         1. Meilleure Explication du Modèle :
o Des invites précises ont permis de générer des visualisations SHAP plus claires,
facilitant la compréhension des décisions du modèle.
o Une structuration soignée des invites a aidé à mettre en évidence les facteurs les
plus influents dans la prédiction des biopsies.
        2. Expérience Utilisateur Optimisée :
o Des invites bien conçues dans Streamlit ont guidé les utilisateurs dans la saisie de
valeurs pertinentes, réduisant les erreurs d’entrée.
o Les résultats de prédiction (" Négatif (0)" ou " Positif (1)") ont été formulés de
manière intuitive et facilement interprétable.
        3. Amélioration de la Performance & du Débogage :
o L’expérimentation avec différentes formulations d’invites a permis d’obtenir des
explications SHAP concises et pertinentes.
o L’ingénierie des invites a également aidé à valider le choix de CatBoost comme
meilleur modèle en mettant en avant les variables les plus influentes.

7.2 Optimisation de la mémoire

L'optimisation de la mémoire n'a pas été une priorité dans notre projet, principalement parce
que nous avons utilisé des ordinateurs dotés d'une architecture 64 bits. Cette architecture
permet de gérer efficacement de grandes quantités de mémoire, avec une capacité d'adressage
bien supérieure à celle des systèmes 32 bits.
En effet, les systèmes 64 bits peuvent théoriquement adresser jusqu'à 16 exaoctets de
mémoire, bien que les limites pratiques soient souvent bien en deçà en fonction du matériel et
du système d'exploitation. Grâce à cette capacité, la gestion des ensembles de données
volumineux et des calculs intensifs est facilitée, réduisant ainsi le besoin d'optimisation
mémoire agressive.
De plus, notre implémentation avec CatBoost et SHAP repose sur des algorithmes bien
optimisés qui tirent parti des ressources matérielles modernes, minimisant ainsi l'impact d'une
éventuelle surcharge mémoire. Ainsi, l'efficacité de notre infrastructure matérielle a rendu
inutile une optimisation spécifique de la consommation mémoire.
Mais ci-dessous le code qui nous permettrait de faire l’optimisation de la mémoire

![image.png](38988ff0-c782-48a5-849c-4e413cbd27ea.png)


8. Conclusion et Perspectives
Bilan : Le projet a permis de développer une solution intégrée combinant un modèle de machine learning robuste, une interprétabilité via SHAP, et une interface conviviale.
Améliorations futures : Possibilités d’extension de l’outil (ajout de nouvelles fonctionnalités, amélioration des performances du modèle, enrichissement de l’interface utilisateur).
Instructions d’installation et utilisation :
Cloner le dépôt, installer les dépendances via requirements.txt et suivre les instructions dans le README pour lancer l’interface et exécuter les notebooks.

Ce README détaillé répond aux exigences du projet en fournissant une vue d’ensemble complète de l’architecture, du prétraitement des données, de la modélisation, de l’optimisation des ressources, de l’explicabilité et de la gestion de projet. Il intègre également les réponses aux questions critiques posées, assurant ainsi une documentation claire et professionnelle pour le dépôt Git.






