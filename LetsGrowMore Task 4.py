#!/usr/bin/env python
# coding: utf-8

# # LetsGrow More: Data Science Internship
# ## Task 4: (Beginner Level Task) Image to pencil sketch with Python
# ### Intern name : Gadade Kailas Rayappa
# ### Step 1 : Importing required modules or libraries

# In[1]:


pip install opencv-python


# In[2]:


import cv2
import matplotlib.pyplot as plt
plt.style.use('seaborn')


# ### Step 2 : Loading original image

# In[3]:


image = cv2.imread("C:\\Users\\Kailas\\OneDrive\\Desktop\\Lets Grow More\\Miss Universe.jpg")
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8,8))
plt.imshow(image)
plt.axis("off")
plt.title("Original Image")
plt.show()


# ### Step 3 : Converting image to grey scale

# In[4]:


image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(8,8))
plt.imshow(image_gray,cmap="gray")
plt.axis("off")
plt.title("GrayScale Image")
plt.show()


# ### Step 4 : Inverting the grey scale image for better details

# In[5]:


image_invert = cv2.bitwise_not(image_gray)
plt.figure(figsize=(8,8))
plt.imshow(image_invert,cmap="gray")
plt.axis("off")
plt.title("Inverted Image")
plt.show()


# In[6]:


image_smoothing = cv2.GaussianBlur(image_invert, (21, 21),sigmaX=0, sigmaY=0)
plt.figure(figsize=(8,8))
plt.imshow(image_smoothing,cmap="gray")
plt.axis("off")
plt.title("Smoothened Image")
plt.show()


# ### Step 5 : Converting image to pencil sketch

# In[7]:


sketch = cv2.divide(image_gray,255-image_smoothing, scale=255)
plt.figure(figsize=(8,8))
plt.imshow(sketch,cmap="gray")
plt.axis("off")
plt.title("Pencilsketch Image")
plt.show()


# ### Thank you ...
