# StreamRHF Implementation

This is a Python implementation of the StreamRHF algorithm for anomaly detection in data streams.
Github Repository link: https://github.com/tarcisiobueno/stream-rhf

The code is based on the following papers:

- STREamRHF: Tree-Based Unsupervised Anomaly Detection for Data Streams by Nesic, Stefan and Putina, Andrian and Bahri, Maroua and Huet, Alexis and Navarro, Jose Manuel and Rossi, Dario and Sozio, Mauro. 
- Random Histogram Forest for Unsupervised Anomaly Detection by Putina, Andrian and Sozio, Mauro and Rossi, Dario and Navarro, José Manuel.

To use it you need to have the following libraries installed:
- numpy
- pandas
- scikit-learn
- tqdm
- joblib

Have the data in ../data/public/. (You may change this path in the code)
Have a folder ./results/ to save the results. (You may change this path in the code)

to run the code use the following command:
python streamRHF.py dataset1 dataset2 dataset3 ...

where dataset1, dataset2, dataset3, ... are the names of the datasets to be processed. They shuld be in the ../data/public/ folder.

We also integrated this implementation to the CapyMOA library. Here is the link to the github repository:

https://github.com/YannTab/CapyMOA
