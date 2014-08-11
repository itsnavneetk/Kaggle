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
            if float(passenger[5]) > 1.9:
                survived_list.append([passenger[0],0])
                continue
            #Males in third class die
            if passenger[1] == "3":
                if float(passenger[8]) > 40:
                    survived_list.append([passenger[0],1])
                    continue
                survived_list.append([passenger[0],0])
                continue
            #Males in second class
            if passenger[1] == "2":
                #Second class males with no age listed die
                if passenger[4] == "":
                    survived_list.append([passenger[0],0])
                    continue
                #Second class males under 14 survive
                if float(passenger[4]) < 13:
                    survived_list.append([passenger[0],1])
                    continue
                survived_list.append([passenger[0],0])
                continue
            #Males in First class
            if passenger[1] == "1":
                #No age listed die
                if passenger[4] == "":
                    survived_list.append([passenger[0],0])
                    continue

                #Fare over 350 live
                if float(passenger[8]) > 350:
                    survived_list.append([passenger[0],1])
                    continue

                #18 and under live
                if float(passenger[4]) < 19:
                    survived_list.append([passenger[0],1])
                    continue

                if float(passenger[4]) > 42:
                    survived_list.append([passenger[0],0])
                    continue

                if float(passenger[8]) < 15:
                    survived_list.append([passenger[0],0])
                    continue

                if passenger[9] != "":
                    if passenger[9][0] == "E":
                        survived_list.append([passenger[0],1])
                        continue

                if float(passenger[8]) < 40:
                    survived_list.append([passenger[0],1])
                    continue



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
                if float(passenger[5]) > 2.2:
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
                if float(passenger[5]) > 0.5:
                    survived_list.append([passenger[0],0])
                    continue
                if float(passenger[4]) > 33:
                    survived_list.append([passenger[0],0])
                    continue
                #All others 3rd class women live (women from "S" between ages 20 and 32)
                survived_list.append([passenger[0],1])
                continue



    return survived_list


print(len(test))

#Call the prediction function on test and write the result to CSV
with open('titanic_submission_14.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(predict_survival(test[1:]))
