# sentiment_analysis
Sentiment analysis using Pytorch and STT datasset

Specify the number of epochs, the batch size, the network, the optimizer and its parameters :


```    
from sentiment import Sentiment
from models import OneLayerNetwork, ConvNetwork

s = Sentiment(n_epochs=10, batch_size=200)
s.train_model(ConvNetwork,
              optimizer=optim.Adam,
              optim_param={'lr' : 10e-3})

```




Result in showing the train and test curves (MSE loss and accuracy)
