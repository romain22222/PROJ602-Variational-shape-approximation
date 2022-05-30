# Variational Shape Approximation
## Qu'est ce que la variational shape approximation ?
Il s'agit d'une approximation de formes à l'aide d'une approche variationnelle. 
Cette méthode d'approximation de maillage procède par minimisation d'une erreur entre le maillage d'orginie et n ensemble de plans.
Pour obtenir la minimisation, il est nécessaire d'appliquer un algorithme de clustering sur les triangles du maillage original, alternant partitionnement et meilleure approximation locale.
On utilise également la déviation des normales introduites pour les appliquer au rendu final.

## Notre code : 
Notre code permet d'utiliser cette approximation sur une figure .obj choisi. 
Ainsi, un affichage se présente à vous, permettant de déterminer les différe,tes zones approximées.

### Son déroulement :
L'algorithme s'exécute n fois (n paramètre choisi)
	récupération des informations de chaque région actuelle (barycentre, normale, les faces qui la compose)

	écupération des faces qui représentent au mieux chaque région (barycentre et normale proches de ceux de la région) -> nommées faces graines
	(cas où l'on vient de commencer : toutes les faces représentent leur propre région)

	création d'une file contenant les faces adjacentes aux faces graines

	assignation de chaque face de la file la région qui représente le mieux celle-ci

	parmi les régions assignées :
        on rajoute 3 régions : 
            une qui sera composé des deux régions avec le moins d'erreur,
            les deux autres sont les deux parties de régions qui font partie de la pire région

	on récupère ensuite pour chaque région les régions qui lui sont adjacentes

	et ensuite on va regarder pour rajouter des régions qui sont des fusion de régions adjacentes qui sont très similaires

	enfin, on reconstruit les nouvelles mailles à partir des régions récupérées

 ### Différents choix : 
1. Obtenir les graines de chaque région
2. Réassigner à chaque face la région qui correspond le mieux
3. Réarranger les régions (réduire la pire, tenter de créer de plus grandes régions)
4. Reconstruire les mailles


## Installations :
### Numpy : 
Pour installer numpy, il vous suffit de faire la commande suivante : 
    python -m pip install numpy

### Polyscope :
Pour installer polyscope, il vous suffit de faire la commande suivante : 
    python -m pip install polyscope


## Exécution du code : 
Pour exécuter le code, vous devez ...


## Différentes illustrations et applications : 
... cf captures ecran 

## Variation des paramètres et leur influence :
.. si on fait varier n 
