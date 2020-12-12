import cv2
import numpy as np

possibilidades_cor=2
preto_e_branco=True
zoom = (150,200)
floyd = cv2.imread('t2.jpeg')
frames=[]
writer = cv2.VideoWriter("floyd.mp4", cv2.VideoWriter_fourcc(*"xvid"), 30,(2*floyd.shape[1],floyd.shape[0]))
if preto_e_branco: 
 floyd = cv2.cvtColor(floyd,cv2.COLOR_BGR2GRAY)
 floyd = cv2.cvtColor(floyd,cv2.COLOR_GRAY2BGR)
for i in range(1,floyd.shape[0]-1):
 for j in range(1,floyd.shape[1]-1):
  for k in range(3-2*preto_e_branco):
   err = floyd[i,j,k]-int((possibilidades_cor)*floyd[i,j,k]//255)*(255/(possibilidades_cor-1))
   floyd[i,j,k] = int((possibilidades_cor)*floyd[i,j,k]//255)*(255/(possibilidades_cor-1))
   floyd[i,j+1,k] += err * (7/16)
   floyd[i+1,j,k] += err * (5/16)
   floyd[i+1,j-1,k] += err * (3/16)
   floyd[i+1,j+1,k] += err * (1/16)
 concat = np.concatenate((floyd[:,:,0],cv2.resize(floyd[max(i-(zoom[1]-zoom[0])//2,0):min(i+(zoom[1]-zoom[0])//2,floyd.shape[0]-1),zoom[0]:zoom[1],0],(floyd.shape[1],floyd.shape[0]),interpolation = cv2.INTER_AREA)), axis=1) if preto_e_branco else np.concatenate((floyd,cv2.resize(floyd[max(i-(zoom[1]-zoom[0])//2,0):min(i+(zoom[1]-zoom[0])//2,floyd.shape[0]-1),zoom[0]:zoom[1]],(floyd.shape[1],floyd.shape[0]),interpolation = cv2.INTER_AREA)), axis=1)
 writer.write(cv2.cvtColor(concat,cv2.COLOR_GRAY2BGR))
 cv2.imshow('floyd',concat);cv2.waitKey(1)
writer.release()
cv2.imwrite('out.png',floyd[:,:,0]) if preto_e_branco else cv2.imwrite('out.png',floyd)