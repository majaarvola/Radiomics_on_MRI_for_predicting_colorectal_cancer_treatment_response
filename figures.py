import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def create_confusion_matrix(yTrue, yPred):
    cf_matrix = confusion_matrix(yTrue, yPred)

    categories = ['Small', 'Major', 'Complete']
    sns.heatmap(cf_matrix, annot=True, xticklabels=categories, yticklabels=categories, cmap='Blues')
    
    plt.xlabel('Predicted label') 
    plt.ylabel("True label") 
    plt.show()
