import numpy as np
import pandas as pd

from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily, Hourly
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

def fetch_data(lat, longti, year, month, day):
    start = datetime(year, month, day)
    end = datetime(year, month, day, 23, 59)
    vancouver = Point(lat, longti)
    data = Hourly(vancouver, start, end)
    return data.fetch()

def split_time(inp):
    day, hour = inp.split(" ")[:2]
    year, month, perday = day.split("-")
    hour, minute, _ = hour.split(":")
    return int(year), int(month), int(perday), int(hour)

def extract_feature(inp):
    timeData = inp[0]
    lat = float(inp[1])
    longti = float(inp[2])
    y, m, d, _ = split_time(timeData)
    return y, m, d, lat, longti


if __name__ == '__main__':
    # import dataset: only the park datasets
    sample_df = pd.read_csv("/datasets/merged_data_complete.csv")

    # insert the date time
    timeData = sample_df["gps_timestamp"].apply(lambda x: split_time(x))
    years = [i[0] for i in timeData]
    months = [i[1] for i in timeData]
    days = [i[2] for i in timeData]
    hours = [i[3] for i in timeData]
    sample_df["year"] = years
    sample_df["month"] = months
    sample_df["day"] = days
    sample_df["hour"] = hours

    # get the weather datasets
    weather_data = []
    for index, pdf in tqdm(sample_df.groupby(["name", "year", "month", "day"])):
        row = pdf.iloc[0, :]
        year, month, day, lat, longti = extract_feature(row)
        sampleOne = fetch_data(lat, longti, year, month, day)
        weather_data.append(sampleOne)

    weather_df = pd.concat(weather_data)
    weather_df = weather_df.reset_index()

    weather_df["year"] = weather_df["time"].apply(lambda x: x.year)
    weather_df["month"] = weather_df["time"].apply(lambda x: x.month)
    weather_df["day"] = weather_df["time"].apply(lambda x: x.day)
    weather_df["hour"] = weather_df["time"].apply(lambda x: x.hour)

    # merge data
    new_merge_df = sample_df.merge(weather_df, on = ["year", "month", "day", "hour"])

    # set up the label
    new_merge_df["label"] = new_merge_df["NO2_ugm3"].apply(lambda x: 0 if x > 40 else 1)

    used_feature = ['latitude', 'longitude','year', 'month', 'day', 'hour', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'coco', 'label']

    new_data = new_merge_df[used_feature].drop_duplicates()

    used_features = ['latitude', 'longitude','year', 'month', 'day', 'hour', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'wpgt', 'pres', 'coco']

    for col in tqdm(used_features):
        mean_values = new_data[col].mean()
        new_data[col].fillna(mean_values, inplace = True)

    sort_new = new_data.sort_values(by = ["year", "month", "day"])
    # split by time
    length = sort_new.shape[0]
    num_test = int(length * 0.1)
    train, test = sort_new.iloc[:-num_test, :], sort_new.iloc[-num_test:, :]

    trainX = train[used_features]
    trainY = train["label"]

    testX = test[used_features]
    testY = test["label"]
    # create decision tree model
    clf = DecisionTreeClassifier()

    clf = clf.fit(trainX.values,trainY)
    y_pred = clf.predict(testX.values)

    print("Accuracy:",metrics.accuracy_score(testY, y_pred))


     