# Domain-Indentification
NLP Applications (Spring, 2019) Course Project.  


NLA_Final_Report.pdf is the Report for this project.

Domain_Identification_Presentation.pptx is for the Presentation for this project.

Data is present in data/ folder.

Outputs can be seen in the notebook as explained below, or even in the report.

Create a directory 'pretrained_embeds/' in the same directory as this repo. Download glove embeddings from http://nlp.stanford.edu/data/glove.6B.zip Unzip it and place file 'glove.6B/' in 'pretrained_embeds/' directory.


## Running Attention Model

Please note for all the code to run smoothly, make sure pre-trained glove embeddings are placed at the right place as mentioned above.

attention/ folder contains all the code for attention based keyword extraction model.

Run the Test_input.ipynb to get the outputs for a given article(example outputs can also be seen here).

Run attention/train.py file to train the model if required with any other dataset.

Trained model is saved in the same attention/ directory  with the file name attention_model.pt.

model.py inside attention/ has the code for the definition of the attention model.

Paramteres like embedding size, hidden state size, can be changed from trian.py file.



