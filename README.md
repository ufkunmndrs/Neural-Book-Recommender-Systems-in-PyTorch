# Neural-Book-Recommender-Systems-in-PyTorch

This project was the final submission for the "Introduction to Neural Networks and Sequence-to-Sequence Learning" class at Ruprecht-Karls-University of Heidelberg, Summer Term 2020 <br>
<br>
The aim of this project was to implement different Book Recommendation Systems and compare them with each other.<br>
The "classic" book recommender system was written in plain Python, while all three neural recommenders were implemented in <b>Pytorch</b>.
All neural recommenders were trained, validated and tested on the <b>goodbooks 10k</b> dataset. The book recommender in plain python was mainly based on *A Programmer's Guide to Datamining - The Ancient Art of the Numerati* by Ron Zacharski. The bulk of the code used for the classic recommender was taken directly from the website and the chapters of Zacharski's page: http://guidetodatamining.com/

## First Steps 
The following steps are required in order to recreate the project
### Prerequisites ✔️
Following modules, versions and Datasets are necessary in order to succesfully run the programs:
- Python 3.7 or higher
- Pytorch 1.6 or higher
- Pandas
- Numpy
- Matplotlib
- Sklearn
- codecs
- goodbooks-10k
- BX-Book-Crossings (optional) 

### Installing 🔥
As most modules are already provided by Anaconda distribution, only Pytorch needs to be installed directly via:<br>
`pip install torch`
or alternatively via<br>
`pip install torchvision`

The goodbooks10k dataset can be downloaded via this link on kaggle:
https://www.kaggle.com/zygmunt/goodbooks-10k

Alternatively, you can attain the dataset by downloading the repository

### Data Preparation 📚

The data is being preprocessed accordingly in the `load_goodbooks_ds.py` file via Pandas module. The plain recommender was expanded in order to load and store the goodbooks-10k dataset, in addition to BX-Book-Crossings. The `load_goodbooks_ds.py` file already provides the train-, validation- and testsplit accordingly and is being imported into all necessary files.  

### Running, training and using the models 🏃

You can train, validate and test the neural models by running the files in the `NeuralModels` - here the Neural Network Recommender as an example:

`python3 src/NeuralModels/NNrecommender.py`

Executing this command will train, validate and test the models and you can then either conduct an evaluation or use the trained recommender.
The plain recommender can be used via executing the following command:

`python3 src/recommender_full.py`

This will create a `recommender` object within the file and load the goodbooks-10k data into it. Alternatively, you can also overwrite the goodbooks-10k data and load the **BX-Book-Crossings** data instead if you desire to. The `recommender` is equipped with every functionality that Zacharski presents on his website, but was also expanded with some other recommendation methods.


### Evaluation 📊

The evaluation of all three neural models was conducted via the methods provided in `metrics.py` and the usual evaluation metrics provided by sklearn's "metrics" module. To access and view those evaluations, you only need to navigate to the `Evaluation` folder and open the Jupyter Notebook file of the model whose evaluation you want to see. 
 

### Acknowledgements

Special thanks go out to my course instructor Michael Staniek, who not only held a brilliant and informative Neural Networks course but also helped me out whenever I encountered problems.
