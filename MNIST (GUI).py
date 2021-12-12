from tensorflow.python.keras.models import load_model
import pygame
from pygame.locals import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
pygame.init()

def Draw():

    model = load_model('DMM.h5')

    while True:
        mouse = pygame.mouse.get_pressed()
        m_event = pygame.event.poll()

        if m_event.type==pygame.QUIT:
            break

        if mouse[0]:
            m_x,m_y = pygame.mouse.get_pos()
            pygame.draw.rect(Screen,(255,255,255),(m_x,m_y,50,50))

        if m_event.type==pygame.MOUSEBUTTONDOWN and m_event.button==BUTTON_RIGHT:
            pixel_array = []
            for i in range(hight):
                for j in range(width):
                    pixel_array.append(Screen.get_at((j,i))[0])

            img_array = np.array(pixel_array)
            img_array = img_array.reshape((width, hight, 1))

            path = 'mnist.jpg'
            cv2.imwrite(path, img_array)

            img = cv2.imread(path, 0)
            resize_img = cv2.resize(img, (28, 28))
            # plt.imshow(resize_img)
            # plt.show()
            array_img = np.asarray(resize_img)
            array_img = array_img / 255

            index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            val = np.argmax(model.predict(array_img.reshape(1, 28, 28, 1)))
            print('PRICTED NUMBER IS : ', index[int(val)])
            Screen.fill((0,0,0))

        pygame.draw.rect(Screen,(0,0,0),(0,0,0,0))
        pygame.display.update()


#-----------------------------------------------------------------------------------------------#

width = 500
hight = 500
Screen = pygame.display.set_mode((width,hight))

Draw()


