import cv2
import os
import torch
import openpifpaf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

def exportImage(frame, points, filename, path="./"):
    if(type(frame) == str):
        frame = cv2.imread(frame)
    
    os.chdir(path)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for i in range(0, len(points)):
        draw(frame, points[i].data, i)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.imwrite(path + "/" + filename + ".png", frame)

    return

def drawTextBoxes(frame, points, index):
    for i in range(0, len(points)):
        if points[i][0] > 0.0 and points[i][1] >0.0:
            color = ""
            text = ""
            if i == 0:
                color = (int("88", 16), int("03", 16), int("FC",16)) # Nariz
                text = "NOSE "

            elif i == 1:
                color = (int("FC", 16), int("88", 16), int("03",16)) # Olho Esquerdo
                text = "EYE_L "

            elif i == 2:
                color = (int("FC", 16), int("88", 16), int("03",16)) # Olho Direito
                text = "EYE_R "

            elif i == 3:
                color = (int("B3", 16), int("6A", 16), int("32",16)) # Orelha Esquerda
                text = "EAR_L "

            elif i == 4:
                color = (int("B3", 16), int("6A", 16), int("32",16)) # Orelha Direita
                text = "EAR_R "

            elif i == 5:
                color = (int("75", 16), int("D1", 16), int("C8",16)) # Ombro Esquerdo
                text = "SHO_L "

            elif i == 6:
                color = (int("75", 16), int("D1", 16), int("C8",16)) # Ombro Direito
                text = "SHO_R "

            elif i == 7:
                color = (int("3A", 16), int("93", 16), int("BD",16)) # Cotovelo Esquerda
                text = "ELB_L "

            elif i == 8:
                color = (int("3A", 16), int("93", 16), int("BD",16)) # Cotovelo Direita
                text = "ELB_R "

            elif i == 9:
                color = (int("34", 16), int("54", 16), int("E3",16)) # Pulso Esquerdo
                text = "WRI_L "

            elif i == 10:
                color = (int("34", 16), int("54", 16), int("E3",16)) # Pulso Direito
                text = "WRI_R "

            elif i == 11:
                color = (int("B5", 16), int("3C", 16), int("7C",16)) # Quadril Esquerdo
                text = "HIP_L "

            elif i == 12:
                color = (int("B5", 16), int("3C", 16), int("7C",16)) # Quadril Direito
                text = "HIP_R "
            
            elif i == 13:
                color = (int("E3", 16), int("32", 16), int("47",16)) # Joelho Esquerdo
                text = "KNE_L "

            elif i == 14:
                color = (int("E3", 16), int("32", 16), int("47",16)) # Joelho Direito
                text = "KNE_R "

            elif i == 15:
                color = (int("00", 16), int("FF", 16), int("AA",16)) # Pé Esquerdo
                text = "ANK_L "

            else:
                color = (int("00", 16), int("FF", 16), int("AA",16)) # Pé Direito
                text = "ANK_R "

            cv2.rectangle(
                frame, 
                (points[i][0]+5, points[i][1]-5), 
                (points[i][0]+5+50, points[i][1]-5-20), 
                color, 
                -1)

            cv2.putText(
                frame,
                text.replace(" ", ""),
                (points[i][0]+5,points[i][1]-5-5), 
                cv2.FONT_HERSHEY_DUPLEX, 
                0.5,
                (255,255,255),
                1,
                cv2.LINE_AA)
    
    return

def drawPoints(frame, points):
    j = 0

    for i in points:
        color = ""

        if j == 0:
            color = (int("88", 16), int("03", 16), int("FC",16)) # Nariz

        elif j == 1:
            color = (int("FC", 16), int("88", 16), int("03",16)) # Olho Esquerdo

        elif j == 2:
            color = (int("FC", 16), int("88", 16), int("03",16)) # Olho Direito

        elif j == 3:
            color = (int("B3", 16), int("6A", 16), int("32",16)) # Orelha Esquerda

        elif j == 4:
            color = (int("B3", 16), int("6A", 16), int("32",16)) # Orelha Direita

        elif j == 5:
            color = (int("75", 16), int("D1", 16), int("C8",16)) # Ombro Esquerdo

        elif j == 6:
            color = (int("75", 16), int("D1", 16), int("C8",16)) # Ombro Direito

        elif j == 7:
            color = (int("3A", 16), int("93", 16), int("BD",16)) # Cotovelo Esquerda

        elif j == 8:
            color = (int("3A", 16), int("93", 16), int("BD",16)) # Cotovelo Direita

        elif j == 9:
            color = (int("34", 16), int("54", 16), int("E3",16)) # Pulso Esquerdo

        elif j == 10:
            color = (int("34", 16), int("54", 16), int("E3",16)) # Pulso Direito

        elif j == 11:
            color = (int("B5", 16), int("3C", 16), int("7C",16)) # Quadril Esquerdo

        elif j == 12:
            color = (int("B5", 16), int("3C", 16), int("7C",16)) # Quadril Direito
        
        elif j == 13:
            color = (int("E3", 16), int("32", 16), int("47",16)) # Joelho Esquerdo

        elif j == 14:
            color = (int("E3", 16), int("32", 16), int("47",16)) # Joelho Direito

        elif j == 15:
            color = (int("00", 16), int("FF", 16), int("AA",16)) # Pé Esquerdo

        else:
            color = (int("00", 16), int("FF", 16), int("AA",16)) # Pé Direito

        if(i[0] > 0.0 and i[1] > 0.0):
            cv2.circle(frame, (i[0], i[1]), 5, color, -1)

        j += 1
    return

def drawJoints(frame, points):
    color = (int("3C", 16), int("3C", 16), int("3C", 16))
    if points[3][0] > 0.0 and points[3][1] > 0.0 and points[1][0] > 0.0 and points[1][1] > 0:
        cv2.line( frame, (points[3][0], points[3][1]), (points[1][0], points[1][1]), color, 5)

    if points[1][0] > 0.0 and points[1][1] > 0.0 and points[0][0] > 0.0 and points[0][1] > 0:
        cv2.line( frame, (points[1][0], points[1][1]), (points[0][0], points[0][1]), color, 5)

    if points[0][0] > 0.0 and points[0][1] > 0.0 and points[2][0] > 0.0 and points[2][1] > 0:
        cv2.line( frame, (points[0][0], points[0][1]), (points[2][0], points[2][1]), color, 5)

    if points[2][0] > 0.0 and points[2][1] > 0.0 and points[4][0] > 0.0 and points[4][1] > 0:
        cv2.line( frame, (points[2][0], points[2][1]), (points[4][0], points[4][1]), color, 5)

    if (points[5][0] > 0.0 and points[5][1] > 0.0) or (points[6][0] > 0.0 and points[6][1] > 0):

        if points[3][0] > 0.0 and points[3][1] > 0.0 and points[5][0] > 0.0 and points[5][1] > 0:
            cv2.line( frame, (points[3][0], points[3][1]), (points[5][0], points[5][1]), color, 5)

        if points[4][0] > 0.0 and points[4][1] > 0.0 and points[6][0] > 0.0 and points[6][1] > 0:
            cv2.line( frame, (points[4][0], points[4][1]), (points[6][0], points[6][1]), color, 5)

        if points[3][0] == 0.0 and points[3][1] == 0.0 and points[4][0] == 0.0 and points[4][1] == 0 and points[6][0] > 0.0 and points[6][1] > 0 and points[5][0] > 0.0 and points[5][1] > 0 and points[0][0] > 0.0 and points[0][1] > 0.0:
            cv2.line( frame, (points[0][0], points[0][1]), (points[5][0], points[5][1]), color, 5)

            cv2.line( frame, (points[0][0], points[0][1]), (points[6][0], points[6][1]), color, 5)

        if points[7][0] > 0.0 and points[7][1] > 0.0 and points[5][0] > 0.0 and points[5][1] > 0:
            cv2.line( frame, (points[7][0], points[7][1]), (points[5][0], points[5][1]), color, 5)

        if points[8][0] > 0.0 and points[8][1] > 0.0 and points[6][0] > 0.0 and points[6][1] > 0:
            cv2.line( frame, (points[8][0], points[8][1]), (points[6][0], points[6][1]), color, 5)

        if points[11][0] > 0.0 and points[11][1] > 0.0 and points[5][0] > 0.0 and points[5][1] > 0:
            cv2.line( frame, (points[11][0], points[11][1]), (points[5][0], points[5][1]), color, 5)

        if points[12][0] > 0.0 and points[12][1] > 0.0 and points[6][0] > 0.0 and points[6][1] > 0:
            cv2.line( frame, (points[12][0], points[12][1]), (points[6][0], points[6][1]), color, 5)

        if points[5][0] > 0.0 and points[5][1] > 0.0 and points[6][0] > 0.0 and points[6][1] > 0:
            cv2.line( frame, (points[5][0], points[5][1]), (points[6][0], points[6][1]), color, 5)

    if (points[7][0] > 0.0 and points[7][1] > 0.0) or (points[8][0] > 0.0 and points[8][1] > 0):

        if points[7][0] > 0.0 and points[7][1] > 0.0 and points[9][0] > 0.0 and points[9][1] > 0:
            cv2.line( frame, (points[7][0], points[7][1]), (points[9][0], points[9][1]), color, 5)

        if points[8][0] > 0.0 and points[8][1] > 0.0 and points[10][0] > 0.0 and points[10][1] > 0:
            cv2.line( frame, (points[8][0], points[8][1]), (points[10][0], points[10][1]), color, 5)

    if (points[11][0] > 0.0 and points[11][1] > 0.0) or (points[12][0] > 0.0 and points[12][1] > 0):

        if points[11][0] > 0.0 and points[11][1] > 0.0 and points[13][0] > 0.0 and points[13][1] > 0:
            cv2.line( frame, (points[11][0], points[11][1]), (points[13][0], points[13][1]), color, 5)

        if points[12][0] > 0.0 and points[12][1] > 0.0 and points[14][0] > 0.0 and points[14][1] > 0:
            cv2.line( frame, (points[12][0], points[12][1]), (points[14][0], points[14][1]), color, 5)

        if points[11][0] > 0.0 and points[11][1] > 0.0 and points[12][0] > 0.0 and points[12][1] > 0:
            cv2.line( frame, (points[11][0], points[11][1]), (points[12][0], points[12][1]), color, 5)

    if (points[13][0] > 0.0 and points[13][1] > 0.0) or (points[14][0] > 0.0 and points[14][1] > 0):
        if points[15][0] > 0.0 and points[15][1] > 0.0 and points[13][0] > 0.0 and points[13][1] > 0:
            cv2.line( frame, (points[15][0], points[15][1]), (points[13][0], points[13][1]), color, 5)

        if points[16][0] > 0.0 and points[16][1] > 0.0 and points[14][0] > 0.0 and points[14][1] > 0:
            cv2.line( frame, (points[16][0], points[16][1]), (points[14][0], points[14][1]), color, 5)
    return

def draw(frame, points, index, text=0):

    pointsToDraw = []

    for i in range(0, len(points)):
        pointsToDraw.append([int(points[i][0]), int(points[i][1]) , points[i][2]])

    drawJoints(frame, pointsToDraw)
    if(text != 0):
        drawTextBoxes(frame, pointsToDraw, index)
    drawPoints(frame, pointsToDraw)

    return

def main():
    img = cv2.imread("images/IMG_1558.jpg")

    scale_percent = 25 #' percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    print(dim)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30')
    predictions, gt_anns, image_meta = predictor.numpy_image(image)
    #print(predictions[0].data)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    exportImage(image, predictions, 'imagemPose')

    imgplt = plt.imread('imagemPose.png')
    plt.imshow(imgplt)
    plt.show()
    

main()
