import numpy as np
import imageio
from skimage.io import imread
from skimage.io import imsave
from skimage.io import imshow
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.color import rgba2rgb
from skimage import img_as_ubyte
from skimage.morphology import dilation, erosion, disk,rectangle
import cv2
from skimage.filters import threshold_otsu




IN1=imread('ir1.jpg')
R1 = IN1[:,:,0]
G1 = IN1[:,:,1]
B1 = IN1[:,:,2]
imshow(B1)
plt.title('Image 1')
plt.show()

IN2=imread('ir2.jpg')
R2 = IN2[:,:,0]
G2 = IN2[:,:,1]
B2 = IN2[:,:,2]

imshow(IN2)
plt.title('Image 2')
plt.show()

# # L'intersection entre FORME1 et FORME2       
# imshow(np.minimum(IN1,IN2))
# plt.title('Intersection des deux images')
# plt.show()

# # L'union entre FORME1 et FORME2   
# imshow(np.maximum(IN1,IN2))
# plt.title('Union des deux images')
# plt.show()


#Boucle de l'intersection de la dilatation de l'intersection entre FORME1 et FORME2 et de l'érosion de l'union entre FORME1 et FORME2
def mediane(FORME1, FORME2):

    INTER = np.minimum(FORME1, FORME2)
    UNION = np.maximum(FORME1, FORME2)
    E=INTER
    TMP=INTER
    i=1
    while np.max(TMP>100) :
        TMP=erosion(UNION,disk(i))
        E=np.maximum(E,np.minimum(TMP,dilation(INTER,disk(i))))
        i=i+i 
    return E


#Insertion des médianes dans une liste pour les couleurs en Rouge
m1R = mediane(R1,R2)
Med_setR = []  

Med_setR.append(R1)
Med_setR.insert(1, m1R)
Med_setR.append(R2)

#Insertion des médianes dans une liste pour les couleurs en Vert
m1G = mediane(G1,G2)
Med_setG = []  

Med_setG.append(G1)
Med_setG.insert(1, m1G)
Med_setG.append(G2)

#Insertion des médianes dans une liste pour les couleurs en Bleu
m1B = mediane(B1,B2)
Med_setB = []  

Med_setB.append(B1)
Med_setB.insert(1, m1B)
Med_setB.append(B2)

def listeMediane(liste, NbMed):
    for i in range (NbMed):
        cpt = 0
        for ind in range(len(liste)-1):
            m1 = liste[ind+cpt] #Récupération de la première médiane
            m2 = liste[ind+1+cpt] #Récupération de la deuxième médiane
            med = mediane(m1, m2) #Calcul de la médiane résultat de m1 et m2
            liste.insert(ind+1+cpt, med) #Ajout de la médiane entre les deux autres médianes
            cpt += 1
            imshow(med)
            plt.show()
    return liste


ensembleImageR = listeMediane(Med_setR, 4)
ensembleImageG = listeMediane(Med_setG, 4)
ensembleImageB= listeMediane(Med_setB, 4)

ensembleImage = []

for r,g,b in zip(ensembleImageR,ensembleImageG,ensembleImageB):
    couleur = np.zeros((200,200,3),dtype=np.uint8)
    couleur[:,:,0] = b
    couleur[:,:,1] = g
    couleur[:,:,2] = r
    ensembleImage.append(couleur)
    
# #Ajouter l'inverse de la liste pour afficher la transformation de la forme dans le sens opposé
# for form in reversed(ensembleImage):
#     ensembleImage.append(form)
    
#imageio.mimsave("NIVEAU.gif",ensembleImage,fps=20)

video = cv2.VideoWriter("trans.mp4",cv2.VideoWriter_fourcc(*'mp4v'),10,(200,200))
for j in range(len(ensembleImage)):
    video.write(ensembleImage[j])

cv2.destroyAllWindows()
video.release()