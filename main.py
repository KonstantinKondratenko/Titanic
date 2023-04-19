import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.ensemble import RandomForestClassifier


class passanger:
    def __init__(self, id, is_survived, pclass, sex, age, embarked):
        self.id = id
        self.is_survived = is_survived
        self.pclass = pclass
        self.sex = sex
        self.age = age
        self.embarked = embarked


class Data_set:
    def __init__(self, id, is_survived, pclass, sex, age, embarked):
        self.data_dict = {}
        self.id_list = id
        self.survived_list = is_survived
        self.pclass_list = pclass
        self.sex_list = sex
        self.age_list = age
        self.embarked_list = embarked


def solve(train, test):
    y = train["Survived"]

    features = ["Pclass", "Age", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(train[features])
    X_test = pd.get_dummies(test[features])

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    predictions = model.predict(X_test)

    output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
    output.to_csv('./Data/model_+_age.csv', index=False)


def draw_2D_graphics_of_one_param(title: str, ox: list, oy: list, x_label: str, y_label: str, label_description: str,
                                  output_file: str):
    plt.figure()
    plt.grid()
    plt.title(title)
    # plt.plot(survived_list, age_list, label=label_description)
    plt.scatter(ox, oy, label=label_description)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(output_file)
    plt.close()


def check_results():
    kagle = pd.read_csv('./Data/gender_submission.csv')
    model = pd.read_csv('./Data/model_without_age.csv')
    model_with_age = pd.read_csv('./Data/model_+_age.csv')
    id_list = kagle["PassengerId"].values.tolist()
    survived_list_kagle = kagle["Survived"].values.tolist()
    survived_list_m = model["Survived"].values.tolist()
    survived_list_ma = model_with_age["Survived"].values.tolist()
    couter_1 = 0
    couter_2 = 0
    couter_3 = 0
    couter_4 = 0
    for i in range(len(id_list)):
        if survived_list_kagle[i] == survived_list_m[i] == survived_list_ma[i]:
            # print(f"{id_list[i]} correct modelling")
            couter_1 += 1
        elif survived_list_kagle[i] == survived_list_m[i]:
            # print(f"{id_list[i]} good result for modeling without age")
            couter_2 += 1
        elif survived_list_kagle[i] == survived_list_ma[i]:
            # print(f"{id_list[i]} good result for modeling with age")
            couter_3 += 1
        elif survived_list_kagle[i] != survived_list_m[i] and survived_list_kagle[i] != survived_list_ma[i]:
            # print(f"{id_list[i]} bad model result")
            couter_4 += 1

    print(f"Counters : {couter_1}-{couter_2}-{couter_3}-{couter_4}")
    print(f"Both models gave the same correct result {couter_1} times")
    print(f"Model without age gave the correct result when model with age gave wrong result {couter_2} times")
    print(f"Model with age gave the correct result when model without age gave wrong result {couter_3} times")
    print(f"Both models gave the same wrong result {couter_4} times")


if __name__ == "__main__":

    df = pd.read_csv('./Data/gender_submission.csv')
    test = pd.read_csv('./Data/test.csv')
    train = pd.read_csv('./Data/train.csv')

    age_mean = train["Age"].mean()
    print("mean value of all ages {",age_mean,"} used to replace nan to this value")
    # remove_nans(age_mean)
    train_without_nans_in_age = pd.read_csv('./Data/train.csv')
    ages_list_mode = train_without_nans_in_age["Age"].values.tolist()
    train_without_nans_in_age["Age"] = train_without_nans_in_age["Age"].fillna(29)
    test["Age"] = test["Age"].fillna(29)

    id_list = train["PassengerId"].values.tolist()
    survived_list = train["Survived"].values.tolist()
    pclass_list = train["Pclass"].values.tolist()
    sex_list = train["Sex"].values.tolist()
    age_list = train["Age"].values.tolist()
    # sib_list = train["SibSp"].values.tolist()
    # parch_list = train["Parch"].values.tolist()
    # ticket_list = train["Ticket"].values.tolist()
    # fare_list = train["Fare"].values.tolist()
    # cabin_list = train["Cabin"].values.tolist()
    embarked_list = train["Embarked"].values.tolist()

    for i in range(len(age_list)):
        if math.isnan(age_list[i]):
            # print(i)
            age_list[i] = age_mean

    for i in range(len(embarked_list)):
        if type(embarked_list[i]) is not str:
            embarked_list[i] = "S"
    # print(id_list)
    # print(survived_list)
    # print(pclass_list)
    # print(sex_list)
    # print(age_list)
    # # print(sib_list) # hard to find correlation
    # # print(parch_list) # hard to find correlation
    # # print(ticket_list) # hard to interpretative
    # # print(fare_list) # too many nan values
    # # print(cabin_list) # too many nan values
    # print(embarked_list)

    # init
    ds = Data_set(id=id_list, is_survived=survived_list, pclass=pclass_list, sex=sex_list,
                  age=age_list, embarked=embarked_list)
    for i in range(len(id_list)):
        human = passanger(id=id_list[i], is_survived=survived_list[i], pclass=pclass_list[i], sex=sex_list[i],
                          age=age_list[i], embarked=embarked_list[i])
        ds.data_dict[id_list[i]] = human
    solve(train_without_nans_in_age, test)
    check_results()
    # draw_2D_graphics_of_one_param('Graphic', survived_list[:100], age_list[:100], "Is survived", 'Age', 'graphic',
    #                               './Data/model')
