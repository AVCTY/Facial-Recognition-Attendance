#!/usr/bin/python3

import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import numpy as np
import cv2
import os

def process_full_labels(directory):
  """List the dir names and directly return"""
  distinct_labels = os.listdir(directory)
  full_classes = []
  for label in distinct_labels:
    for id in list(range(0, len( os.listdir(f"{directory}/{label}") ))):
      full_classes.append(label)
      
  return full_classes


def process_distinct_labels(directory):
  """List the dir names and directly return"""
  distinct_labels = os.listdir(directory)
  full_classes = []
  for label in distinct_labels:
    full_classes.append(label)
      
  return full_classes

def labels_str_to_int(labels):
  """Turn the string into an int, the 'n' will be removed"""
  int_labels = []
  for label in labels:
    int_labels.append( int( label[1:] ) )

  return int_labels

def read_images(dirname, distinct_labels, verbose=""):
  """Does an imread and returns the whole data"""
  all_read_img = []
  index = 0
  for partial_path in distinct_labels:
    
    partial_path = f"{dirname}/{partial_path}"
    images_names = os.listdir( partial_path )
    
    for single_image in images_names:
      full_path = f"{partial_path}/{single_image}"
      # print(f"Prepare to read {full_path}")
      #if verbose and not index % 1000:
      #  print (f"Reading {index}/{ len( distinct_labels ) }...")
      read_img = cv2.imread( full_path )
      read_img = cv2.resize( read_img, (28, 28) )

      processed_img = read_img / 255
      processed_img = processed_img.flatten() # don't flatten to view img
      all_read_img.append(read_img)
      #plt.imshow( all_read_img[0] )
      #plt.show()


  return all_read_img

# besiyata d'shmaya
