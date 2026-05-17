# Projet Convex Optimization - Parcours A

## Sujet

Application de techniques d'optimisation a un probleme concret de Machine Learning : classification du niveau de popularite de morceaux Spotify a partir de caracteristiques musicales.

Dataset : Spotify 1 Million Tracks, environ 1,1 million de morceaux et une vingtaine de colonnes.

Problematique proposee :

> Dans quelle mesure un reseau de neurones peut-il classer les morceaux Spotify selon leur niveau de popularite a partir de leurs caracteristiques musicales, et quel optimiseur permet l'entrainement le plus efficace ?

---

## 1. Plan complet du notebook

1. Import des librairies
2. Chargement du dataset `spotify_data.csv`
3. Exploration rapide : dimensions, types, valeurs manquantes, statistiques descriptives
4. Visualisations avant modelisation
5. Nettoyage : doublons, colonnes inutiles, valeurs manquantes
6. Creation de `popularity_class`
7. Separation `X` / `y`
8. Split train / validation / test stratifie
9. Encodage de `genre`
10. Normalisation des variables numeriques
11. Creation du reseau neuronal
12. Experience avec SGD
13. Experience avec SGD momentum
14. Experience avec Adam
15. Experience optionnelle avec RMSprop
16. Comparaison des courbes d'apprentissage
17. Evaluation sur le test set
18. Matrice de confusion et rapport de classification
19. Interpretation des resultats
20. Conclusion

Le fichier [spotify_popularity_nn_optimizers.py](./spotify_popularity_nn_optimizers.py) est structure en cellules `# %%` et peut etre ouvert comme notebook dans VS Code ou converti en notebook Jupyter.

---

## 2. Structure conseillee du rapport final, 10 a 20 pages

| Section | Contenu attendu | Longueur indicative |
| --- | --- | ---: |
| 1. Introduction | Contexte, objectif, interet du dataset Spotify | 1 page |
| 2. Parcours A et optimisation | Role de l'optimisation dans le Machine Learning | 1 page |
| 3. Presentation du dataset | Colonnes, types de variables, cible, interet | 1-2 pages |
| 4. Problematique | Question principale et sous-questions | 0,5 page |
| 5. Pretraitement | Nettoyage, encodage, normalisation, split | 2 pages |
| 6. Tache ML | Transformation de `popularity` en classes | 1 page |
| 7. Reseau neuronal | Architecture, activations, perte, regularisation | 1-2 pages |
| 8. Optimiseurs | SGD, momentum, Adam, RMSprop optionnel | 2-3 pages |
| 9. Experiences | Tableau des hyperparametres et protocole | 1-2 pages |
| 10. Resultats | Courbes, tableaux, matrice de confusion | 2-3 pages |
| 11. Convexite / non-convexite | Discussion mathematique et consequences | 2 pages |
| 12. Limites | Popularite influencee par des facteurs externes | 1 page |
| 13. Conclusion | Reponse a la problematique et ameliorations | 1 page |
| 14. Bibliographie | Kaggle, Keras/TensorFlow, articles optimiseurs | 0,5 page |
| 15. Annexes | Code, tableaux complets, parametres | variable |

---

## 3. Presentation du dataset

Le dataset Spotify 1 Million Tracks regroupe environ 1,1 million de morceaux. Chaque ligne correspond a un titre musical et contient des informations textuelles, des metadonnees et des caracteristiques audio calculees par Spotify.

Variables quantitatives :

`popularity`, `year`, `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`, `time_signature`

Variables qualitatives :

`artist_name`, `track_name`, `track_id`, `genre`

Choix de colonnes :

| Colonne | Decision | Justification |
| --- | --- | --- |
| `track_id` | Supprimee | Identifiant technique sans signification musicale directe |
| `track_name` | Supprimee | Texte non exploite sans NLP |
| `artist_name` | Supprimee dans la version de base | Tres forte cardinalite, encodage direct couteux |
| `genre` | Conservee | Variable musicale importante, encodable par one-hot |
| `popularity` | Transformee puis supprimee des features | Sert a construire la cible ; la garder causerait une fuite de donnees |

---

## 4. Choix de la tache Machine Learning

La tache retenue est une classification multiclasse :

> Predire le niveau de popularite d'un morceau Spotify a partir de ses caracteristiques musicales.

La variable `popularity`, initialement numerique entre 0 et 100, est transformee en trois classes :

| Classe | Intervalle | Interpretation |
| ---: | --- | --- |
| 0 | 0 a 30 | Faible popularite |
| 1 | 31 a 70 | Popularite moyenne |
| 2 | 71 a 100 | Forte popularite |

Extrait de code :

```python
def popularity_class(score):
    if score <= 30:
        return 0
    elif score <= 70:
        return 1
    else:
        return 2

df["popularity_class"] = df["popularity"].apply(popularity_class)
```

Cette formulation est adaptee a un reseau neuronal, car elle permet de combiner des variables numeriques normalisees et une variable categorielle encodee.

---

## 5. Pretraitement des donnees

Etapes :

1. Charger le dataset avec `pandas`.
2. Supprimer la colonne d'index eventuelle `Unnamed: 0`.
3. Verifier les dimensions et les types.
4. Identifier les valeurs manquantes.
5. Supprimer les doublons exacts.
6. Creer `popularity_class`.
7. Supprimer `popularity` des variables explicatives.
8. Supprimer `track_id`, `track_name` et `artist_name` pour la version de base.
9. Imputer les valeurs numeriques par la mediane.
10. Imputer les valeurs categorielles par la modalite la plus frequente.
11. Encoder `genre` par One-Hot Encoding.
12. Normaliser les variables numeriques avec `StandardScaler`.
13. Faire un split stratifie train / validation / test.

Point important : le scaler et l'encodeur doivent etre appris uniquement sur le train set, puis appliques au validation set et au test set.

---

## 6. Reseau neuronal utilise

Architecture de reference :

| Couche | Taille | Activation |
| --- | ---: | --- |
| Entree | nombre de features apres preprocessing | - |
| Dense | 128 | ReLU |
| Dropout | 0.2 | - |
| Dense | 64 | ReLU |
| Dropout | 0.2 | - |
| Dense | 32 | ReLU |
| Sortie | 3 | Softmax |

Justifications :

- ReLU est simple, efficace et limite le probleme de gradients tres faibles par rapport a sigmoid ou tanh.
- Softmax transforme les sorties finales en probabilites sur les trois classes.
- La cross-entropy est adaptee a la classification multiclasse.
- Le dropout et la regularisation L2 limitent le surapprentissage.
- L'early stopping evite de continuer l'entrainement lorsque la validation loss ne s'ameliore plus.

---

## 7. Theorie des optimiseurs

### SGD

La descente de gradient stochastique met a jour les poids a partir d'un mini-batch :

```text
w_{t+1} = w_t - eta * grad L(w_t)
```

`eta` est le learning rate. SGD est simple, peu couteux en memoire et adapte aux grands datasets. En revanche, il depend fortement du learning rate et peut converger lentement.

### SGD avec momentum

Le momentum ajoute une memoire de la direction precedente :

```text
v_{t+1} = beta * v_t + grad L(w_t)
w_{t+1} = w_t - eta * v_{t+1}
```

Il reduit les oscillations et accelere souvent la convergence dans les directions coherentes du gradient.

### Adam

Adam combine une idee de momentum et une adaptation individuelle du pas d'apprentissage pour chaque parametre. Il estime :

- un premier moment du gradient, proche d'une moyenne mobile ;
- un second moment, proche d'une moyenne mobile des gradients au carre.

Adam converge souvent vite sur les reseaux neuronaux et demande moins de reglage initial que SGD. Il peut cependant donner une generalisation differente de SGD selon le probleme.

### RMSprop

RMSprop adapte le learning rate a partir d'une moyenne mobile des gradients au carre. Il est utile lorsque les gradients ont des echelles tres differentes selon les parametres.

---

## 8. Convexite et non-convexite

L'entrainement d'un reseau neuronal est generalement un probleme non convexe. La fonction de perte depend des poids de plusieurs couches composees avec des activations non lineaires. Cette composition cree un paysage d'optimisation complexe.

Consequences :

- il peut exister plusieurs minima locaux ;
- il peut exister des points selles ;
- l'initialisation des poids influence la trajectoire d'optimisation ;
- le learning rate peut accelerer, ralentir ou destabiliser l'apprentissage ;
- il n'existe pas de garantie simple d'atteindre un optimum global ;
- les optimiseurs stochastiques cherchent surtout une bonne solution empirique.

Ce point est central pour le parcours A : le projet ne consiste pas seulement a obtenir une accuracy finale, mais a analyser le comportement des methodes d'optimisation pendant l'entrainement.

---

## 9. Tableau des experiences a realiser

| Experience | Optimiseur | Learning rate | Batch size | Architecture | Dropout |
| ---: | --- | ---: | ---: | --- | ---: |
| 1 | SGD | 0.01 | 512 | 128-64-32-3 | 0.2 |
| 2 | SGD momentum | 0.01 | 512 | 128-64-32-3 | 0.2 |
| 3 | Adam | 0.001 | 512 | 128-64-32-3 | 0.2 |
| 4 | RMSprop | 0.001 | 512 | 128-64-32-3 | 0.2 |
| 5 | Adam | 0.0001 | 512 | 128-64-32-3 | 0.2 |
| 6 | Adam | 0.001 | 256 | 128-64-32-3 | 0.2 |

---

## 10. Tableau de resultats a remplir

| Experience | Optimiseur | LR | Batch | Epochs avant arret | Temps (s) | Best val loss | Val accuracy | Val F1 macro | Test accuracy | Test F1 macro | Commentaire |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | SGD | 0.01 | 512 |  |  |  |  |  |  |  |  |
| 2 | SGD momentum | 0.01 | 512 |  |  |  |  |  |  |  |  |
| 3 | Adam | 0.001 | 512 |  |  |  |  |  |  |  |  |
| 4 | RMSprop | 0.001 | 512 |  |  |  |  |  |  |  |  |
| 5 | Adam | 0.0001 | 512 |  |  |  |  |  |  |  |  |
| 6 | Adam | 0.001 | 256 |  |  |  |  |  |  |  |  |

Le script genere automatiquement `outputs/optimizer_comparison_results.csv`.

---

## 11. Visualisations a produire

Avant modelisation :

- distribution de `popularity` ;
- distribution de `popularity_class` ;
- distribution des genres ;
- matrice de correlation des variables numeriques ;
- relation entre `popularity_class` et certaines variables audio.

Pendant l'entrainement :

- train loss et validation loss ;
- train accuracy et validation accuracy ;
- comparaison des optimiseurs ;
- nombre d'epochs avant early stopping.

Apres prediction :

- matrice de confusion ;
- rapport de classification ;
- F1-score par optimiseur ;
- performance finale sur le test set.

---

## 12. Phrases d'analyse a adapter aux resultats

Pour la convergence :

> L'optimiseur [NOM] atteint une validation loss faible plus rapidement que [NOM]. Cela indique une convergence plus rapide dans les premieres epochs, probablement due a [l'adaptation du learning rate / l'effet du momentum].

Pour SGD :

> SGD presente une convergence plus lente et plus sensible au learning rate. Ce comportement est coherent avec la theorie, car l'algorithme applique un meme pas d'apprentissage global a tous les parametres.

Pour SGD momentum :

> L'ajout du momentum ameliore la stabilite par rapport a SGD simple. Les oscillations de la validation loss sont reduites et le modele atteint plus rapidement une zone de bonne performance.

Pour Adam :

> Adam obtient une convergence rapide, ce qui est coherent avec son mecanisme d'adaptation du learning rate par parametre et l'utilisation de moments du gradient.

Pour le surapprentissage :

> Lorsque la loss d'entrainement continue de diminuer alors que la validation loss stagne ou augmente, le modele commence a surapprendre. Le dropout, la regularisation L2 et l'early stopping permettent de limiter ce phenomene.

Pour les limites du dataset :

> Les caracteristiques audio ne suffisent pas a expliquer toute la popularite d'un morceau. La popularite depend aussi de facteurs externes comme la notoriete de l'artiste, la promotion, les playlists, les tendances, le pays et la periode de sortie.

Pour la non-convexite :

> Les differences observees entre optimiseurs illustrent la non-convexite du probleme. Chaque methode suit une trajectoire differente dans l'espace des parametres et peut atteindre une solution locale differente.

---

## 13. Conclusion scientifique type

Ce projet a permis d'appliquer des techniques d'optimisation a un probleme concret de Machine Learning : la classification de morceaux Spotify selon leur niveau de popularite. Le reseau neuronal construit exploite des caracteristiques audio numeriques et le genre musical pour predire trois classes de popularite.

Les experiences montrent que le choix de l'optimiseur influence fortement la vitesse de convergence, la stabilite de l'apprentissage et les performances de validation. SGD constitue une reference simple mais sensible au learning rate. L'ajout du momentum ameliore generalement la trajectoire d'optimisation. Adam converge souvent plus rapidement grace a l'adaptation du pas d'apprentissage et aux moments du gradient.

Sur le plan de l'optimisation, l'entrainement du reseau neuronal correspond a un probleme non convexe. Il n'est donc pas possible de garantir l'atteinte d'un optimum global. L'objectif pratique est plutot d'obtenir une solution suffisamment bonne et stable. Les resultats dependent de l'initialisation, du learning rate, du batch size, de l'architecture et du choix de l'optimiseur.

Enfin, les performances doivent etre interpretees avec prudence. La popularite Spotify depend de nombreux facteurs absents du dataset : marketing, playlists, notoriete de l'artiste, tendances, pays, promotion et reseaux sociaux. Les caracteristiques audio apportent donc une information utile mais incomplete.

Ameliorations possibles :

- encoder `artist_name` avec une strategie adaptee a la forte cardinalite ;
- utiliser des embeddings pour `genre` et `artist_name` ;
- integrer les paroles avec des methodes de NLP ;
- tester une classification binaire ;
- comparer d'autres architectures ;
- realiser une recherche plus systematique des hyperparametres.

---

## 14. Bibliographie indicative

- Dataset Kaggle : Spotify 1 Million Tracks.
- Documentation TensorFlow / Keras : optimizers, callbacks, neural networks.
- Documentation scikit-learn : preprocessing, train-test split, metrics.
- Kingma, D. P. and Ba, J. Adam: A Method for Stochastic Optimization.
- Goodfellow, I., Bengio, Y., Courville, A. Deep Learning.
