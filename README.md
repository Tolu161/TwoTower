# TwoTower
Document Search : CNNs and the Two Towers - Building a search engine which takes in quieries and produces a list of relevant document . 

![image](https://github.com/user-attachments/assets/64e9ca7e-cfc7-4e9b-af67-9ae742d9ee01)

1. We start by taking our triple of (query, relevant document, irrelevant document) which should be tokenised. - tokeniser.py
   
2. This is fed into an embedding layer to turn each query and document into a list of its constitutent token embeddings. This embedding layer could be a word2vec style embedding layer which is pretrained, and then its weights are frozen for the downstream task. - triplegenerateandembeddings.py , testgeneratetriplet.py
   
3. We then make separate CNN, MLP encoders for the queries and documents. We feed the query token embeddings to the query encoder to get a query encoding, and each set of document token embeddings to the document encoder to to get a document encoding. These two separate encoders are what give the architecture its name as a Two Tower architecture. The reason for having these two separate encoders is to try to capture the fact that queries and documents tend to have different semantic and syntactic structure, and so may need to be encoded differently. - TowerTower.py
   
4. Finally, we pass the encoded query, relevant document and irrelevant document to the Triplet Loss Function in order to do backpropagation and update the parameters of our two encoders. We will discuss this more in the lesson on Training. - TowerTower.py 

At Inference : 

![image](https://github.com/user-attachments/assets/d725bb0d-6ca4-45ff-8c9d-963035f36ab1)
