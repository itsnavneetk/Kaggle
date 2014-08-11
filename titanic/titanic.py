#Titanic Kaggle comp work
import csv

train = [line for line in csv.reader(open("titanic_train.csv",newline=''))]
test = [line for line in csv.reader(open("titanic_test.csv",newline=''))]

cols = test[0]

#Display column indexes
for i,v in enumerate (cols):
    print(i,v)


#Custom tree to predict survival

def predict_survival(passenger_list):
    survived_list = [["PassengerId","Survived"]]

    for passenger in passenger_list:

        #Split passengers into males and females
        if passenger[3] == "male":
            #Males with high SibSP die
            if float(passenger[5]) > 2:
                survived_list.append([passenger[0],0])
                continue
            #Males in third class die
            if passenger[1] == "3":
                survived_list.append([passenger[0],0])
                continue
            #Males in second class
            if passenger[1] == "2":
                #Second class males with no age listed die unless Fare > 9
                if passenger[4] == "":
                    if float(passenger[8]) < 10:
                        survived_list.append([passenger[0],0])
                        continue
                    else:
                        survived_list.append([passenger[0],1])
                        continue
                #Second class males under 15 survive
                if float(passenger[4]) < 15:
                    survived_list.append([passenger[0],1])
                    continue
                if float(passenger[4]) > 14:
                    survived_list.append([passenger[0],0])
                    continue
            #Males in First class
            if passenger[1] == "1":
                #No age listed die
                if passenger[4] == "":
                    survived_list.append([passenger[0],0])
                    continue
                #18 and under live
                if float(passenger[4]) < 19:
                    survived_list.append([passenger[0],1])
                    continue
                #Fare over 400 live
                if float(passenger[8]) > 400:
                    survived_list.append([passenger[0],1])
                    continue
                #Age over 39 die
                if float(passenger[4]) > 39:
                    survived_list.append([passenger[0],0])
                    continue
                #Fare under 15 die
                if float(passenger[8]) < 15:
                    survived_list.append([passenger[0],0])
                    continue
                #Fare over 150 die
                if float(passenger[8]) > 150:
                    survived_list.append([passenger[0],0])
                    continue
                #Passengers in section E and B live
                if passenger[9] != "":
                    if passenger[9][0] == "E" or passenger[9][0] == "B":
                        survived_list.append([passenger[0],1])
                        continue
                #All other males die
                survived_list.append([passenger[0],0])
                continue
            #All other males die
            survived_list.append([passenger[0],0])
            continue

        #Females
        else:
            #All first and second class females survive
            if passenger[1] == "1" or passenger[1] == "2":
                survived_list.append([passenger[0],1])
                continue

            if passenger[1] == "3":
                #High Sibling third class females die
                if float(passenger[5]) > 2.3:
                    survived_list.append([passenger[0],0])
                    continue
                #Live if no age listed
                if passenger[4] == "":
                    survived_list.append([passenger[0],1])
                    continue
                #Live if age under 20
                if float(passenger[4]) < 20:
                    survived_list.append([passenger[0],1])
                    continue
                #3rd class Women over 19 with SibSp over 0 die
                if float(passenger[5]) > 0:
                    survived_list.append([passenger[0],0])
                    continue
                #High Parch die
                if float(passenger[6]) > 2.5:
                    survived_list.append([passenger[0],0])
                    continue
                #Sections E and G live
                if passenger[9] != "":
                    if passenger[9][0] == "E" or passenger[9][0] == "G":
                        survived_list.append([passenger[0],1])
                        continue

                #Embarked from C live
                if passenger[10] == "C":
                    survived_list.append([passenger[0],1])
                    continue
                #Embarked from Q die
                if passenger[10] == "Q":
                    survived_list.append([passenger[0],0])
                    continue
                #Embarked from S fare over 15 die
                if float(passenger[8]) > 15:
                    survived_list.append([passenger[0],0])
                    continue
                #Embarked from S age over 32 die
                if float(passenger[4]) > 32:
                    survived_list.append([passenger[0],0])
                    continue
                #All others 3rd class women live (women from "S" between ages 20 and 32)
                survived_list.append([passenger[0],1])
                continue



    return survived_list


print(len(test))

#Call the prediction function on test and write the result to CSV
with open('titanic_submission_11.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(predict_survival(test[1:]))
