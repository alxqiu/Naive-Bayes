# Naive-Bayes
Implementation for naive bayes classification algorithm

How To Use
The program takes in command line arguments and user console input to work. First, when
running the program from command line or from an IDE, ensure that you give arguments for the input
csv file, the output text file, and optionally, a random seed number for the partitioning. When giving the
file names, be sure to give the relative path for the file names in the event your data is in a subdirectory
of the directory the python file is in. Otherwise, if the input file exists outside of the directory of your
python file, give the absolute file path name instead.

If running from command line:
  >python naive-bayes.py <input_file_name>.csv <output_file_name>.txt <seed>
  
 After you hit run, the algorithm will create a dataframe object from the csv file, and when it is
finished, it will prompt you with more information on the console. It will display the column names
parsed from the dataframe and ask which column of the data should represent the “classes”, or the
labels for type classification. The code will re-prompt you in the case you do not enter a valid column
name. It is advised to pick a class column identified by the source of the data set, as its possible to use
other columns but the algorithm may not perform well. Once you have given a valid column name, the
code will prompt you through the console to ask if you want to exclude more columns. In general, I
recommend excluding any columns with non-numerical data apart from the class column. If the class
column contains non-numerical entries that is okay, but non-class columns with non-numerical data
should be excluded as they will cause an error with the calculations. For instance, when working with
the Star detection data set, I recommend exclusion of the Star color and Spectral Class columns. To
exclude, respond “Y” to the “Any columns to exclude” question, and type the name of each column you
would like to exclude, and type “done” when you are finished entering in column names.
When complete, the algorithm should say “Printed to <output_file_name>.txt successfully! :)”
when it has finished all tasks, and the results should be visible on that file.


Algorithm Explanation
The naïve bayes algorithm works by using bayes formula to find the probability of a record being
a certain class. This means that for each record, we need it to be able to be in the form of a vector, and
be able to find the probability of P(Cj | x), or the probability of vector x being of class Cj. P(Cj | x) is
essentially the product of P(xi | Cj) for all xi features of vector x, multiplied by P(Cj) or overall probability
of class Cj occurring among the records. P(Cj) is calculated using the overall probability of a given class Cj
occurring among the records in the training set. P(xi | Cj) is calculated using the gaussian probability
density function used to find the probability of xioccurring for a continuous variable in a normal
distribution. We find the normal distribution using mean and standard deviation for a feature within a
certain class during model training, and as the features are partitioned by class, using a probability
density function lets us accurately get the probability of obtaining xi given a class or P(xi | Cj). When each 
P(Cj | x) is calculated for each class Cj, the class to label vector x as is chosen by ranking the natural logs
of each P(Cj | x), then choosing the highest value out of those. This is consistent as a score, as the
natural log is a monotonically increasing function, that preserves the relative order of the highest
probabilities.
During training, we collect separate “buckets” of data for each class, and within each bucket, we
then collect the samples of data for each feature. Once we’ve parsed all the records we are supposed to
use for training, we then build our model by taking the mean and standard deviation of each feature, for
each individual class. This helps us run the probability density function during testing for each feature
and have results for P(xi | Cj) accurately separated by class. The value of P(Cj) for each class Cj is found by
the overall proportion of Cj within the training data set.
Runtime of the algorithm is explained at the end of “Code Design”.


Datasets
Links:
https://www.kaggle.com/uciml/glass
https://www.kaggle.com/deepu1109/star-dataset
Both .csv files are the only option available from these links, and to use them, download the csv
file and put in the same directory as naïve-bayes.py, and run the program with the csv file name
included in the command line arguments. In general, these data sets are a mix of good performance and
low performance given this algorithm but are all good examples of a mix of numerical data and have
options for classification columns. For both, I advise using “Star type” and “Type” as the class columns,
and although the code will still work if you choose something like “Spectral Class” for the class column,
the performance will drop significantly.
For other use, the algorithm needs data sets with a class label column and preferably two to
three features that are numerical data. The algorithm cannot perform the calculations of std. deviation
and mean on non-numerical data, but it’s fine with having the class column be of numerical or
enumeration or string type, provided that the values of a single class are cased consistently.


Code Design
There are multiple stages to the code, which are configuring the program, partitioning, training
the model, testing with test data, scoring the test results, and writing the summary of results.
Configuration of the program involves gathering the correct data needed to perform the algorithm from
the user during runtime, as not all information can be communicated through command line arguments.
The program reads the data from csv to dataframe, and then asks the user for columns to identify as the
class labels and the columns to exclude. Some column names include spaces, and so it would be harder
to have these names communicated through command line arguments. Once the column names are
identified, the algorithm should be good to go, as the input, output, and random seeding information
should have been known before runtime through the command line arguments.
Partitioning uses Python’s random module to separate the dataframe indices into two and can
be seeded for predictable results using the seed command line argument. Partitioning then creates a 
dictionary mapping “test” and “train” to sets of indices, so that the program can cleanly identify which
set of indices it is meant to interact with. Training the model involves parsing through all training indices
and collecting the samples of each feature for each class, and then finding the mean and standard
deviation of each feature for each class.
Testing then involves taking the probability density function on each feature of the test record,
for each class, and finding the product of that and the probability of each class, to make a value for the
probability of each class given the overall record, and then labeling each record based on the highest
natural log of that result. The test suite gives back a dictionary mapping each test record index to a class
label. All class labels will be derived from whatever values were found during training from the class
column identified. Once test results are found, the overall performance, and the detection performance
for each class is identified using Jaccard Similarity, using the sets of indices for each class between the
test results and actual labels. For example, the Jaccard score for a given class K is essentially the
cardinality of the intersection of row index sets of predicted and actual indices belonging to K, divided
by the cardinality of union of those two sets. The overall Jaccard Similarity is the mean of each class’s
jaccard score. The results for each class’s Jaccard Similarity, and the overall similarity are written to the
output file, as well as the probability summaries for each class and each feature found from model
training.
Overall, the algorithm runs in Θ(nmk) time, for n records, m features, and k classes, as the
dominant task of the algorithm is training and testing, which in total reads n records. Finding
distributions and taking a probability from the distributions is Θ(mk) for each record, as each record
needs to have k features be checked for m classes. Therefore, the overall algorithm’s runtime is Θ(nmk).


Pseudocode
Naïve-Bayes (input_file, output_file, seed):
dataframe <-- pd.read_csv(input_file)
(class_col, to_exclude) <-- AskUser()
drop to_exclude from input_file dataframe
(train_indices, test_indices) <-- Partition(dataframe, seed)
train_results <-- train(dataframe, class_col, train_indices)
test_results <-- test(dataframe, class_col, test_indices, train_results)
jaccard_results <-- jaccard(dataframe, class_col, test_results)
format_output(output_file, jaccard_results, test_results, train_results)


Results
Star detection performs extraordinarily well compared to glass data set based on several
factors. For one, characteristics such as temperature and radius tend to have drastic differences even
while keeping star color hidden. For instance, the mean for Brown Dwarfs (class 0) temperature is just
above 3,000 Kelvin, whereas the mean for the same metric for Supergiants (class 4) is almost 16,000
Kelvin. This demonstrates that there is known link between the traits and the classifications, this
algorithm can perform exceptionally well, as the algorithm performance on the star dataset is between
95-98% overall Jaccard Similarity (average Jaccard similarity across classes).
However, as we see on the glass data set, the algorithm performs poorly when features in a
column have closely grouped values and distributions across classes, and individual classes tend to be 
detected well when there is a certain feature, they have sharper differences in mean and standard
deviation in compared to other classes. As a known weakness of the algorithm is how it treats each
feature as independent for each entry, often what influences the classification is how similar other class
distributions are to the true class, which increases the possible value for P(x_i | C) given another class C
that is not the true class that feature x_i in some record belongs to. This in turn increases the likelihood
that class C is chosen incorrectly.


Conclusion
I’ve appreciated how straightforward this algorithm is, but I can identify where is weaknesses lie
based on similarity of data across classes. I think attempting another algorithm such as k-nearest
neighbors and judging the results from the same Jaccard similarity could speak to the ability of one
algorithm over another. I think the algorithm would be best suited to not necessarily identify subtle
patterns from data, but rather used as a quick way to classify a huge sum of data where a general
pattern of differences could be eyeballed. This also helps researchers identify perhaps which features
should be included and excluded to increase this algorithms performance, as that can help mitigate the
downside of having all features be considered independent from each other, which is not accurate in
real life. Perhaps using this data on image recognition data could be useful if we could find a way to
quantify features like we did with star data set, as it could be a fast and simple way to categorize images
that humans don’t have time to process. 
