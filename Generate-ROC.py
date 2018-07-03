################################################################
# Generate ROC plot
################################################################
# input: csv(score[],label[])
# original: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
################################################################
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
# read data to dataframe
df = pd.read_csv('roc_input.csv')

# prepare input
y_test = df[['manual_result']] # test data
y_score = df[['bot_score']] # probability predicted from a classifier

# ROC for binary class
fpr, tpr, thr = roc_curve(y_test, y_score, pos_label=1, drop_intermediate=False)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Bot Detection Result')
plt.legend(loc="lower right")
plt.show()


import json
import os
import pickle
import operator

def load_object(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

df = load_object('G:\\work\\TwitterAPI\\Cleaning\\all_1205-26062018\\df_bot_results.pkl')
bot_user = df[df.bot_score > 0.5][['uid', 'screen_name','bot_score']]
bot_user.head
save_object(bot_user, 'G:\\work\\TwitterAPI\\Cleaning\\all_1205-26062018\\bot-from-bot_score.pkl')