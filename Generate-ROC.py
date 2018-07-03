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
