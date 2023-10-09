import numpy as np 
import matplotlib.pyplot as plt             # importing necessary libraries
import pandas as pd 

df = pd.read_csv("IRIS.csv")


x = df.iloc[: ,:-1].values          # x value as petal length , width...
y = df.iloc[:, 4].values            # y value as flower names



#plt.hist(x)
plt.plot(x,y)
plt.show()



from sklearn.model_selection import train_test_split
x_Train, x_Test, y_Train, y_Test = train_test_split (x,y,test_size = 0.20)   # splitting the test and train value as 20 and 80 percentage



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()                                #fitting the values in between -1 to 1
scaler.fit(x_Train)

x_Train = scaler.transform(x_Train)

x_Test = scaler.transform(x_Test)



from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)                  # using KNN to classify the flower and it sees 5 nearest neighbour
knn.fit(x_Train,y_Train)

#y_pred = knn.predict(x_Test)

#from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
#print(accuracy_score(y_Test,y_pred))                                                  
#print(classification_report(y_Test,y_pred))                                     # Training the dataset
#print(confusion_matrix(y_Test,y_pred))


sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))                    #getting the input from users
petal_width = float(input("Enter petal width: "))

# Standardize the user input using the same scaler
user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])              
user_input_scaled = scaler.transform(user_input)                                  # scaling the users input

# Predict the flower species
predicted_species = knn.predict(user_input_scaled)  # predicting the user input
  
# Display the predicted species
print("Predicted flower species:", predicted_species[0]) 