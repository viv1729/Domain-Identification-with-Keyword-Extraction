# Domain-Indentification
NLP Applications (Spring, 2019) Course Project.  


`NLA_Final_Report.pdf` is the Report for this project.

`Domain_Identification_Presentation.pptx` is for the Presentation for this project.

Data is present in `data/folder`. (This is smaller dataset of ~5k news articles.)

Larger dataset link: [https://drive.google.com/open?id=19XEP1zoZVIyhtglttHgg_Xz9uDB_uXz7]

Outputs can be seen in the notebook as explained below, or even in the report.

Create a directory `pretrained_embeds/` in the same directory as this repo. Download glove embeddings from http://nlp.stanford.edu/data/glove.6B.zip Unzip it and place file `glove.6B/` in `pretrained_embeds/` directory.


## Important Files

`EDA.ipynb`: Some analysis about the dataset.

`Classifier-svm, lr.ipynb`: Contains code for SVM and Logistic Regression Classifier.

`Domain-ClassificationV2.ipynb` Conatains the code for pre-processing, tokenizing, Bi-LSTM model for both English and Hindi (translated). 
                               Go to Prediction section and run `make_prediction` function on an article to predict.
                               
`make_prediction( article='', true_category='', needTranslation=False, verbose=True)`
                               
`Test_Input.ipynb`: Contains code for Bi-LSTM model with attention. Run `make_pred` function for prediction and domain based keyword extraction from an article.


## Attention Model

Please note for all the code to run smoothly, make sure pre-trained glove embeddings are placed at the right place as mentioned above.

Run `attention/train.py` file to train the model if required with any other dataset.

Trained model is saved in the same `attention/` directory  with the file name `attention_model.pt`.

`model.py` inside `attention/` has the code for the definition of the attention model.

Paramteres like embedding size, hidden state size, can be changed from `trian.py` file.

