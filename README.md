# mesure_activite_humaine

Link to datasets: https://drive.google.com/drive/folders/1ruuit4PpnR_DXcTkf8YM3IeuRo0xYCVZ?usp=sharing

## Authors
Hartley Marc - LISTIC (marc.hartley@hotmail.com)

## Documentation complète
Après avoir cloné le projet, ouvrez le fichier "doc/html/index.html" dans un navigateur.

## Processus classique:
Ce projet est utilisé dans le but d'enregistrer de manière automatique une scene à l'aide d'une caméra de profondeur.

Cela permet d'extraire le squelette d'une personne dans une sequence, de les stocker dans une base de données InfluxDB et d'analyser la démarche de la personne.

Les caractéristiques extraites sont:
- La vitesse de la marche,
- La longueur des pas,
- La largeur des pas,
- Le ratio "double-support",
- Le ratio "stance/swing" sur chaque jambe.

Un Raspberry Pi s'occupe d'enregistrer les signaux de plusieurs capteurs sur lequel il est branché, et de stocker cela dans une base de données InfluxDB.

Avec les données des capteurs qui vont servir d'entrée et les caractéristiques de la caméra pour la "verité terrain", nous pouvons entrainer des réseaux de neurones.

Plusieurs types de réseaux de neurones sont proposés ici:
- Un réseau de type MLP (Multi-Layer Perceptron)
- Des réseaux convolutifs:
    - CNN (Convolutional Neural Network) "classique"
    - ResNet (CNN avec l'utilisation de "résidus")
    - Modèles prédéfinis de type AlexNet, GoogleNet, VGG, etc...
  
## Programmes requis
- Python 3.6 ou 3.7 (pas en-dessous pour pouvoir utiliser la skeletisation, pas au-dessus pour pouvoir utiliser le SDK des caméras de profondeur)
- R Studio
- Arduino IDE (ou n'importe quel IDE pour coder sur un Arduino)
- Un GPU compatible CUDA

## Installation du projet
### Sous Linux
- Attention: Le mieux c'est d'utiliser Ubuntu18 car c'est "sûr" de fonctionner. Ubuntu 20.0 est incompatible (à cause de la skeletisation Cubemos).
- Cloner ce repertoire:
```git 
cd ~/Documents
git clone https://github.com/kemar74/mesure_activite_humaine.git
# Renseigner votre GitHub username et password avec lequel vous avez un accès dans ce répertoire
```
- Lancer le script d'installation en suivant les indications affichées:
```bash
sudo bash ~/Documents/mesure_activite_humaine/project_installation.sh
```

### Sous Windows
- Cloner ce repertoire:
```git 
cd ~/Documents
git clone https://github.com/kemar74/mesure_activite_humaine.git
# Renseigner votre GitHub username et password avec lequel vous avez un accès dans ce répertoire
```
- Après, essayez d'installer PyRealSense2 et Cubemos Skeleton Tracker SDK comme vous pouvez
    
## Utilisation du programme:
### Programmation des capteurs
Un programme doit être compilé sur chaque capteur pour être utilisables:
1) Ouvrir un IDE pour coder les Arduino, et ouvrir le fichier "pir_sensors_program.c"
2) Brancher un capteur via la prise USB-C vers l'ordinateur
3) Modifier la variable "`nom_variable`" pour qu'elle s'accorde au numéro indiqué sur le capteur (0x20, 0x21, 0x22 ou 0x23)
4) Compiler vers la carte Arduino.

### Capture des données des capteurs
1) Placer les capteurs tels que vous le souhaitez dans le couloir. 
2) Les brancher au Raspberry Pi (par I2C pour l'instant).
3) (Si vous commencez de zéro :) Placer les fichiers du dossier `files_for_raspberry` (`xiaopir.py`, `continuous_experiment.py` et `hamming_coding.py`) dans le Raspberry (peu importe où, tant qu'ils sont ensemble).
4) **Verifier que la variable `sensors_height` dans le fichier `continuous_experiment.py` soit en accord avec la disposition de vos capteurs (Pour les PIRs, nous avons utilisé "`hips`" lorsque à hauteur de hanches et "`ankles`" lorsque ils sont placés au sol). Cette variable permet d'ajouter un "tag" aux données pour les filtrer/trier par la suite.**
5) Lancer le script `continuous_experiment.py` sur le Raspberry Pi. En cas d'erreur "OSError: 121", verifier que tous les capteurs sont bien branchés (débrancher/rebrancher plusieurs fois).
```cmd
python3 continuous_experiment.py
```

Conditions d'utilisation:
- Le Raspberry doit avoir un accès au serveur Listic (192.168.143.29)
- Dans l'état actuel, il est indispensable de brancher 4 capteurs avec les adresses esclaves "0x20", "0x21", "0x22" et "0x23". (voir "Programmation des capteurs")

Indications:
- À l'heure actuelle, les données transmises des capteurs au Raspberry sont encodées par un codage de Hamming. Le code d'encodage compilé sur les capteurs doit être compatible avec le code de décodage codé sur le Raspberry (fichier `hamming_coding.py`)

### Capture des squelettes par camera de profondeur
1) Placer le cube "Aruco Box" dans le couloir.
2) Placer la caméra dans un coin du dispositif (chercher une croix de scotch en dessous d'un extincteur).
3) Brancher la caméra à l'ordinateur. **Utiliser un cable certifié USB3.0 et brancher sur un port USB 3.0**.
4) Utiliser un environnement virtuel python (voir `create_new_env.txt`)
4) Depuis l'ordinateur, lancer le script `direct_camera_to_db.py` avec la commande CMD suivante :
```cmd
source virtual_env/bin/activate
python direct_camera_to_db.py --remote
```
5) Marcher de la porte vitrée "Nord" jusqu'à l'extincteur "Sud" puis du "Sud" au "Nord" autant de fois que voulu.

### Visualiser une scène
Pour voir le dernier passage réalisé, vous pouvez jouer le script `display_skeleton.py`
```cmd
source virtual_env/bin/activate
python display_skeleton.py
```
Ceci rejoue le dernier enregistrement en base de données. Pour jouer l'avant-dernier, remplacer l'ID de sequence de `-1` à `-2` (`-3` pour l'avant-avant-dernier, etc.).

### Création des datasets pour l'IA
On utilise un dataset different selon le modèle de deep learning à utiliser. 
Les données des datasets sont choisies selon le tag 'sensors_height' qui a été choisi lors de la collecte des données de capteurs (voir "Capture des données des capteurs"). 

**Dans le script `create_pir_csv.py` ET `create_dataset_mlp.R`, modifier la variable `sensors_height` afin qu'elle corresponde aux experimentations que vous souhaitez (ex : "hips" pour les capteurs PIRs à hauteur de hanche, "ankles" quand ils sont au sol, etc.)**
```cmd
source virtual_env/bin/activate
python create_pir_csv.py
R --vanilla < create_dataset_mlp.R
```

Le script Python `create_pir_csv.py` extrait toutes les informations à partir de la base de données et construit un CSV "gait_values_[sensors_height].csv" (avec [sensors_height] celui que vous avez indiqué dans le script). Il crée aussi les données "image" en enregistrant la representation de la Transformée d'Ondelettes Continues (CWT) en image PIL.Image de taille 128×128 dans un fichier "all_pirs_transformed_[sensors_height].data", qui est un object python sauvegardé par "pickle". 

Le script R "create_dataset_mlp.R" va créer le fichier CSV "sequences_resumed_[sensors_height].csv", qui est utilisé par les réseaux de neurones MLP.

### Entrainement de réseaux de neurones
#### Pour entrainer un réseau de type MLP:
- Changer la variable `sensors_height` du fichier `MLP_training.py` pour être en accord avec le dataset créé (`sequences_resumed_[sensors_height].csv`).
- Modifier la variable `dependent_variables` pour entrainer le MLP sur la caractéristique désirée.
- Modifier les paramètres d'entrainement du modèle MLP comme vous le souhaitez (lignes ~114)
- Lancer le script d'entrainement :
```cmd 
source virtual_env/bin/activate
python MLP_training.py
```

#### Pour entrainer un réseau convolutif
- Modifier les paramètres permettant de tester plusieurs structures differentes des CNN avec la variable `models_hyperparameters` (ligne ~1381)
  (nombre de couches convolutives, taille des kernels utilisés, structure des couches cachées finales, fonctions d'activation, etc.)
- Modifier les paramètres d'entrainement (learning rate, optimiseur, type de modèles entrainés) avec la variable `training_parameters` (ligne ~1407)
- Lancer le script d'entrainement :
```cmd 
source virtual_env/bin/activate
python deep_neural_training.py --train --parameter all
```
Cela entraine des réseaux sur tous les paramètres possibles, mais il est possible d'en choisir un spécifique :
- 'body_speed' = vitesse de marche
- 'footstep_width' = largeur de pas
- 'cycle_time' = durée d'un cycle de marche
- 'stride_length' = longueur des pas
- 'corridor_distance' = distance de la personne par rapport au mur (mur Est)

#### Comparer les modèles entrainés
Idem que pour l'entrainement, modifier uniquement l'argument "train" en "display" :
```cmd 
source virtual_env/bin/activate
python deep_neural_training.py --display --parameter all
```

De cette manière un tableau rassemblant les résultats de tous les entrainements de modèles sera affiché, classé par ordre de corrélation des estimations avec la verité terrain. *Ce paramètre peut être modifié par la valeur de la variable `ordering` (`deep_neural_training.py`, ligne ~1491)*

Pour le modèle avec les meilleurs résultats (critère de `ordering`), on affiche les résultats sur un jeu de test de 20% du dataset avec une comparaison entre l'estimation et la verité terrain et des statistiques de l'erreur (en erreur absolue et en pourcentage d'erreur).

Pour visualiser ces affichages avec tous les modèles entrainés, ajouter l'argument `--display-all`
```cmd
source virtual_env/bin/activate
python deep_neural_training.py --display --parameter body_speed --display-all
```

## Quelques avancées (pas forcément fructueuses)
### Capture automatique
*Dans le temps, jadis*, les flux de la caméra étaient stockés en ROSbags (fichiers qui stockent la totalité des flux d'un appareil pour avoir des résultats totalement reproductibles). Ces fichiers sont très lourds, j'ai alors mis en place un système pour lancer le stockage uniquement lorsqu'une personne est dans le champ de vision.

Malheureusement, le processus laisse une latence d'environ une seconde entre la detection et l'enregistrement de la personne, ce qui est trop long pour l'experimentation actuelle.

Dans le doute, je laisse le code du système dans le fichier `automatic_capture.py`, ainsi qu'une démonstration de son utilisation dans le même fichier.

### Reconnaissance faciale
Un petit algorithme de reconnaissance faciale a été utilisé (mais jamais testé...) dans le fichier `person_identification.py`. Une demonstration est disponible en lançant le script `draw_skeleton.py`

Pour information, l'identification se fait par la transformation d'une photo de visage en histogramme HOG. Des SVM sont entrainés pour classifier le HOG d'un nouveau visage parmi le dataset.

### Classification d'actions
Une classifieur d'actions est disponible dans le fichier `train_action_classifier.py`

##  Pour plus de détails sur le travail
Le rapport de stage associé à ce travail est disponible sous le nom de `Rapport_fin_de_stage_Nantes_LISTIC.pdf`.

Pour encore plus d'informations, ne pas hésiter à me contacter.