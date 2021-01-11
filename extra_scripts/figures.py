import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def create_confusion_matrix(yTrue, yPred, categories = ['No', 'Small', 'Major', 'Complete']):
    cf_matrix = confusion_matrix(yTrue, yPred)

    sns.heatmap(cf_matrix, annot=True, xticklabels=categories, yticklabels=categories, cmap='Blues')
    
    plt.xlabel('Predicted label') 
    plt.ylabel("True label") 
    plt.show()


if __name__ == '__main__':
    yTrueTest = [1, 3, 2, 2, 2, 2, 3, 1]
    yPredTest = [2, 1, 1, 2, 2, 2, 2, 2]
    yTrueVal = [1, 2, 2, 2, 3, 2, 2, 0, 1, 1, 2, 2, 3, 2, 3, 3, 2, 2, 1, 3, 2, 1, 2, 1, 2, 1, 2, 3, 1, 3, 2]
    yPredVal = [2, 2, 2, 2, 2, 2, 1, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 3, 2, 2, 2, 2, 2, 2]
    
    create_confusion_matrix(yTrueTest, yPredTest, categories = ['Small', 'Major', 'Complete'])
    create_confusion_matrix(yTrueVal, yPredVal)
