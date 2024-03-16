# handle the system stuff, colab stuff, etc
import os
try:
    from google import colab  # for use in google colab!!
    DIR = os.path.abspath("./drive/My Drive/meta-opt")
except: 
    DIR = os.path.abspath("./")
assert os.path.isdir(DIR)

# make sure we have the necessary folders
for subdir in ['data', 'figs', 'datasets']: 
    temp = os.path.join(DIR, subdir)
    if not os.path.isdir(temp): os.mkdir(temp)
    