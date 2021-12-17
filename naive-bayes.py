import math
import numpy as np
import pandas as pd
import sys
import random


"""
Takes in an input dataframe, and outputs a dictionary in the format:
{
"train" : train_indices
"test" : test_indices
}
Where each indices list in the values are the indices used in
the corresponding key operation (train or test). Indices are randomly 
assigned and split 50-50. 
"""
def partition(input_df):
    """
    Partitions the dataframe with train and test indices through dictionary.
    :param input_df:
    :return: dict{str : set(int)} : Return two sets of integers mapped
            to strings labeled "train" and "test", which should be about
            the same size.
    """
    split = {"train": set(), "test" : set()}

    # generate floor(num_rows / 2) number of training indices
    for i in range(math.floor(len(input_df.index) / 2)):
        # basically do a do-while loop\
        new_idx = random.randint(0, input_df.index.max())
        while new_idx in split["train"]:
            new_idx = random.randint(0, input_df.index.max())
        split["train"].add(new_idx)

    # give test_indices other half
    for i in range(len(input_df.index)):
        if i not in split["train"]:
            split["test"].add(i)
    return split

"""
ask user for exclusion col names and validate accordingly

drop all columns to exclude : mostly non-integer values
"""
def validate_exclusion(input_df, class_column):
    """
    Ask user for which columns to exclude.
    :param input_df: dataframe to parse and validate column choices
    :param class_column: column for class labels.
    :return: input dataframe with columns excluded removed.
    """
    exclude_columns = set()
    yes_set = {"Y", "Yes", "yes", "y"}
    want_to_exclude = (input("Any columns to exclude? (Y/N) ") in yes_set)
    if want_to_exclude:
        print("Type \"done\" when finished")
        counter = 0
        while want_to_exclude:
            user_input = input("Exclusion Column #" + str(counter) + " Name : ")
            if user_input != "done":
                if (user_input not in input_df.columns) or (user_input == class_column):
                    print("Invalid Column: must exist in dataframe and cannot be class column")
                    continue
                exclude_columns.add(user_input)
                counter += 1
            else:
                want_to_exclude = False

    # remove all excluded columns
    # input_df.drop(columns=list(exclude_columns), axis=1)
    for col in exclude_columns:
        del input_df[col]

    return input_df


"""
Get distributions from the records to train. 

Returns summary data in format:
{
"class 1": {"probability": y, "feature 1": {std: a, mean: b}, "feature 2": {std: a_2, mean: b_2}, ...}
"class 2": {"probability": x,"feature 1": {std: a, mean: b}, "feature 2": {std: a_2, mean: b_2}, ...}
}

Where [(stdeviation_1, mean_1), (stdeviation_2, mean_2), ...] correspond to
standard deviation and mean for each feature, given a certain class. 
"""
def train(input_df, class_column, train_indices):
    """
    data_buckets looks like:
    {
        "class_1" : {"feature 1": [1, 2, 3], "feature 2": [1, 2, 3], ...}
        ...
    }
    Where each subarray inside a value represents a collection of
    data for that certain feature.
    :param input_df : dataframe : df with all data from csv
    :param class_column : class : which column is the classification labels
    :param train_indices : set(int) : indices which are for training
    :return:
    """
    data_buckets = {}
    class_counts = {}
    # collect buckets of data for each class, and each feature
    for i in train_indices:
        row_series = input_df.iloc[i]
        class_type = row_series.at[class_column]
        row_series = row_series.drop(labels=class_column)
        if class_type not in data_buckets.keys():
            data_buckets[class_type] = {}
            class_counts[class_type] = 1
            for j in range(row_series.size):
                data_buckets[class_type][row_series.index[j]] = []
        # add to all features buckets for class:
        class_counts[class_type] += 1
        for j in range(row_series.size):
            if row_series.iloc[j] is not None:
                data_buckets[class_type][row_series.index[j]].append(row_series.iloc[j])

    # then process the buckets for mean and standard deviation
    summary = {}
    for class_type in data_buckets.keys():
        summary[class_type] = {}
        summary[class_type]["probability"] = class_counts[class_type] / len(train_indices)
        for feature in data_buckets[class_type].keys():
            # feature is an array in itself
            summary[class_type][feature] = {}
            summary[class_type][feature]["std"] = np.std(data_buckets[class_type][feature])
            summary[class_type][feature]["mean"] = np.mean(data_buckets[class_type][feature])
    return summary


"""
Tests records according to summary data and outputs in format:
{
record_n : classA,
record_m : classB,
record_o : classA,
...
}

For each class probability, do product of all probability of being class given
feature 
"""
def test(input_df, class_column, test_indices, summary):
    """
    Classifies each record in test_indices

    :param input_df: Dataframe : dataframe containing all records
    :param class_column: class : any time representing which column in dataframe is the class
    :param test_indices: set(int) : set of indices for testing
    :param summary: dict() : probability summary from training
    :return: Dictionary mapping record indices to class labels
    """
    tested_records = {}
    for i in test_indices:
        probabilities = {}
        """
        probabilities looks like:
        {classA: 0.56, classB: 0.42,...}
        
        holds P(classA | X), for each class for each 
        vector X (which is row_series). 
        """
        row_series = input_df.iloc[i].drop(labels=class_column)
        for class_type in summary.keys():
            curr_prob = summary[class_type]["probability"]
            for feature in row_series.keys():
                if row_series.at[feature] is not None:
                    # each class is essentially product of all feature given
                    curr_prob *= gaussian_pdf(row_series.at[feature],
                                              summary[class_type][feature]["std"],
                                              summary[class_type][feature]["mean"])
            # doing log instead to make continuous comparison function
            # All log functions between log(0..1) will be negative,
            # so 0 will be the max, so we just let it be neg infinity if curr_prob is 0
            probabilities[class_type] = np.log(curr_prob) if curr_prob != 0 else float('-inf')
        tested_records[i] = max(probabilities, key=probabilities.get)
    return tested_records


def gaussian_pdf(x, sigma, mu):
    """
    Gaussian probability density function, using the continuous
    variable x, distribution std deviation (sigma), and mean (mu).
    :param x: float : value of continuous random variable to check probability of.
    :param sigma: float : standard deviation of given distribution.
    :param mu: float : mean of given distribution.
    :return: Probability of finding value x within the normal distribution
            defined by sigma and mu, for standard deviation and mean, respectively.
    """
    if sigma == 0:
        return 1 if x == mu else 0

    # according to gogle
    # power is the number eulers number is raised by in the gaussian pdf formula
    power = -(math.pow(x - mu, 2) / (2 * math.pow(sigma, 2)))
    return (1 / (sigma * math.sqrt(2 * math.pi))) * (math.pow(math.e, power))


def jaccard_scores(input_df, class_column, test_results):
    """
    Scoring suite used to test accuracy of label classification, by comparing
    the indices for each class predicted by the test suite against the indices
    belonging to each class in the actual labeled results.

    :param input_df: pd.Dataframe : Dataframe object representing source data.
    :param class_column: str : Label of column of dataframe representing classes.
    :param test_results: dict{int : class} : Test record indices mapped to class labels.
    :return: dictionary mapping class labels to scores in format {class : float}
            scores are non-integer numbers in range [0, 1].
    """
    real_sets = {}
    for i in test_results.keys():
        if input_df.iloc[i].at[class_column] not in real_sets.keys():
            real_sets[input_df.iloc[i].at[class_column]] = set()
        real_sets[input_df.iloc[i].at[class_column]].add(i)

    result_sets = {}
    for record in test_results.keys():
        if test_results[record] not in result_sets.keys():
            result_sets[test_results[record]] = set()
        result_sets[test_results[record]].add(record)

    jaccard = {}
    for class_type in set(result_sets.keys()).intersection(set(real_sets.keys())):
        jaccard[class_type] = len(result_sets[class_type].intersection(real_sets[class_type])) \
                              / len(result_sets[class_type].union(real_sets[class_type]))
    return jaccard


def format_output(output_file, jaccard_results, test_results, probability_summary):
    """
    Write to output file with details on jaccard scores on the tested
    records, as well as information on the probability distributions for
    each class per feature in probability summary

    :param output_file: str : name of output file, ending with .txt
    :param jaccard_results: dict{class : float} : Class name keys mapped to
            score values from [0, 1]. Scores are written to output in percentages.
    :param test_results: dict{int : class} : Maps record indices from original
            data set to classification label
    :param probability_summary: dict{class : dict} : Summary of probability
            and random distributions for each feature given a class label.
    :return: None
    """
    # print accuracy results onto output.txt
    with open(output_file, 'w') as file_obj:
        file_obj.write("Records Tested: %d\n" % len(test_results))
        file_obj.write("Jaccard Similarity Overall: %s\n" %
                       "{:.4%}".format(np.mean(list(jaccard_results.values()))))
        file_obj.write("Jaccard Similarity Per Class: \n")
        for class_type in jaccard_results:
            file_obj.write("\tClass %s: %s\n" % (str(class_type),
                        "{:.4%}".format(jaccard_results[class_type])))
        file_obj.write("Statistical Summary Per Class: \n")
        for class_type in probability_summary:
            file_obj.write("\tClass %s : {Probability : %s" % (str(class_type), "{:.4%}".format(
                probability_summary[class_type]["probability"])))
            for field in probability_summary[class_type]:
                if (field == "probability"):
                    continue
                file_obj.write(", %s : %s" % (field, str(probability_summary[class_type][field])))
            file_obj.write("}\n")


"""
Command Line Run: python naive-bayes.py data.csv output.txt"

Then interact with user through console or cmd.
"""
if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception("Please give at least two arguments after naive-bayes.py in format: "
                        "naive-bayes.py <input.csv> <output.txt> <random seed>")
    if not sys.argv[1].endswith(".csv"):
        raise Exception("Please give valid .csv file as input.")
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    if len(sys.argv) > 3:
        random.seed(int(sys.argv[3]))

    # validate that class column name is consistent with the file
    input_df = pd.read_csv(input_file)
    print("Input Column Names: " + str(input_df.columns))
    print("Please give name of column for class labels (all input is case sensitive): ")
    class_column = input()
    while class_column not in input_df.columns:
        class_column = input("Please give valid class column name: ")
    # run exclusion function
    input_df = validate_exclusion(input_df, class_column)
    # run partition on test and train data
    indices = partition(input_df)
    # finish building probabilities from training
    # mapping keys (classes) to values (probability distributions per feature)
    probability_summary = train(input_df, class_column, indices["train"])
    # run test suite on the data
    test_results = test(input_df, class_column, indices["test"],
                        probability_summary)
    # check jaccard similarity and format output to output file
    jaccard_results = jaccard_scores(input_df, class_column, test_results)
    format_output(output_file, jaccard_results, test_results, probability_summary)
    print("Printed to %s successfully! :)" % output_file)


