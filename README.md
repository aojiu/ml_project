# Prediction of Apple Stock price using Social consensus & Relevant Financial Info
Machine learning 2019 spring project

## Object
Using news, exchange rate, ralevence stock to pridict the Apple's stock price change

## General method
Collecting the news from 2011 to 2017. Then using sentiment analysis to evaluate the articals. Next processing the features from news analysis into the same days. Meanwhile, we have collected data contains exchange rate, ralevence stocks price, and Apple stock price. Now we have to merge news, exchange rate, ralevence stocks price with same day as features. Apple stock price is target. For each day that stock price and if it's prior day also has the stock price, then this day target would have features value of prior day. However, if it's prior day has no stock price, the feature would be average of all priors no stock days' features.  <br>

## Dataset Description and Pre-processing:

### News Dataset
#### Original News Dataset:
url : https://www.kaggle.com/snapcrack/all-the-news#articles3.csv <br>
In this dataset, there could be many articals in the same days. <br>

#### Features:
Title, publication, author, date, year, month, url, content <br>
We only use the content to do the sentiment analysis, and keep the date only. 

#### Pre-processing News Dataset:
##### Sentiment Analysis:
We use the Empath sentiment anaylsis.<br>
Github: https://github.com/Ejhfast/empath-client<br>
The sentiment analysis will analyze a string type of artical, and return a value of how ralevence to the target word.
###### example:
	lexicon.analyze("he hit the other person", normalize=True)
	# => {'help': 0.0, 'office': 0.0, 'violence': 0.2, 'dance': 0.0, 'money': 0.0, 'wedding': 0.0, ... )
We use part of this return values as features for news. <br>
##### Features for creating new news dataset
There three groups features. <br>
First group contains features that show how much a artical is relevant to the Apple Company<br>
######
	"social_media","computer","business","programming","hearing","urban",
	"shopping","science","work","valuable","fashion","technology","competing","economics","office"
	
Second group contains features that show the negative emotions<br>
#####
	'hate','aggression','horror','suffering','ridicule','irritability',
	'deception','disappointment','negative_emotion','nervousness'
Third group contains features that show the positive emotions<br>
#####
	'cheerfulness','optimism','celebration','trust','positive_emotion'
Next we merge these values for each article to each day since there many articles for one day.<br>
##### Separate into two datasets
Now we have a dataset with three group features, but we want process more. Thus we create three new features, which are<br>
######
	relevance, positive_relevance, negative_relevance
relevance = avg(relevant group features value)<br>
positive_relevance = relevance * avg(positive group features value)<br>
negative_relevance = relevance * avg(negative group features value)<br>
The reason for creating three features is that we want to explore the relevant articals with such emotions in general would have any relation with the stock price. <br>
As result, we have two dataset. One contains all three group features, and Other only contains three features.

### Exchange Rate Dataset:
https://github.com/datasets/exchange-rates/tree/master/data
#### Features:
	Euro exchange rate, Hong Kong exchange rate, Canada exchange rate, China exchange rate
As we consider that Apple is global company, so when exchange rate changes, the sales rate toward to other countries would changes as well, which may influence the stock prices. 

### Relevant Stock Price:
#### Features:
	QCOM, TXN, MSFT
Did some researches on which public companies are related to APPLE and include some of them in our dataset.<br>
Qualcomm: https://finance.yahoo.com/quote/QCOM/history/ <br>
Texas Instrument: https://finance.yahoo.com/quote/TXN/history/ <br>
Microsoft: https://finance.yahoo.com/quote/MSFT/history/ <br>


### APPLE Stock price as Y:
https://finance.yahoo.com/quote/AAPL/history/ <br>
It is the common sense that we use the n day's news to predict n+1 day's stock price. <br>
And there are two situations that we need to consider:<br>
<br>
The day has stock price. Under such circumstance, we just need need to consider last day's news. Because n - 2 day's effects on the stock market have applied on the n - 1 day. So, we just get the stored news information from n - 1, and use it to predict n day's stock price.<br>
<br>
The other situation is that the day does not have stock price. This may happen during weekend and holidays. We will store all the consective non-stock day's news information, and wait until the first day where there is stock price. Then we take all the stored information and normalize the information. The reason for doing this is that we need to consider all possible effects of news during the non-stock days. Any of them has the possibiity to affect the next stock price day.<br>
<br>

### Creat Two diffent typs y:
for different model, we need different target value. Such as that we need continuous number for linear regression, but we need binary number for SVM and Neural Network models.<br>
As result, the target for continuous number is close price of each day for apple and the difference between close price and open price of day. The target for discrete number is when close price - open price less than 0, or more than 0.

## Linear Regression:
we used Polynomial Linear Regression with Ridge <br>
We use 1 - 5 degree transformation for features, with Ridge of 0, 0.35, 0.5 and 0.75 for each degree. 

## SVM
first we process the target dataset values with difference of open price and close price in a day into 1 and -1. <br>
Next we apply three different kernal, linear, RBF, and Poly to two datasets with all the news parameters and only three news parameters. <br>
Meanwhile, we also use the pca to increase the runtime and proformance of different kernal's SVM models

## Neural Network 
We set up the Keras and TensorFlow for building the model. <br>
For this two dataset, we change the target value to 1 and 0. If difference price is less than 0, then it is 0, otherwise it is 1. <br>
For each datasets, the models would apply many different parameters. <br>
#### hidenLayer : 
Hidden layer as list. eg [10,4,2] is three hiden layer with 10 neurons in first hiden layer, 4 neurons in second layer, and 2 neurons in last layer etc. <br>
#### activation : 
Activation function. eg. elu, relu, sigmoid, hard_sigmoid, exponential, linear <br>
#### activity_regularizer: 
We want regulariz the value after caculating the activation function<br>
#### optimizer:  
Optimiz function that used to update the parameters each generation. eg. SGD, RMSprop, Adagrad, Adadelta, Adam<br>
#### batch_size: 
Number of samples per gradient update. <br>
#### epochs: 
Number iterations of generation update. <br>

#### Function:
	def nn_classifier(x, y, hidenLayer, activation, regularizer,  optimizer, batch_size, epochs):
	    model = Sequential()
	    n_parameter = x.shape[1]
	    model.add(Dense(hidenLayer[0], input_shape = (n_parameter,), activation = activation, name = "hidden",
			   activity_regularizer = regularizers.l2(regularizer)))

	    if len(hidenLayer) > 1:
		for layer in range(1,len(hidenLayer)):
		    name_in = "hidden" + str(layer)
		    model.add(Dense(hidenLayer[layer], activation = activation, name = name_in,
			   activity_regularizer = regularizers.l2(regularizer)))

	    model.add(Dense(1, activation=activation,activity_regularizer = regularizers.l2(regularizer), 
			    name='output'))

	    model.compile(optimizer=optimizer,
			  loss='binary_crossentropy',
			  metrics=['accuracy'])

	    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

	    model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size,verbose=0)

	    lossi_Out, AccOut = model.evaluate(X_test,Y_test, verbose=0)
	    lossi_In, AccIn = model.evaluate(X_train,Y_train, verbose=0)

	    model.summary()
    

## Performance 
### Linear Regression:
According to our model selection, the model that yields the best MSE performence for news emotion relevance is:<br>
Degree two polynomial transfermation and 0.35 ridge regularization. <br>
The best model for all features dataset is: <br>
Degree two polynomial transfermation and 0.35 ridge regularization. <br>
### SVM:
The Kernel methodd that yields highest testing accuracy is: <br>
Linear kernel with c = 0.6 - 0.8 approximately. The accuracy is around 72% - 75%.<br>
### Neural Network
The model that yields the highest accuracy for new emotion relevance dataset is: <br>
optimizer = rmsprop, activation = relu, batch_size = 130, epoch = 1000, regularizer = 0.001
which yields in accuracy = 0.77186 and out sample accuracy = 0.71212 <br>
<br>
The model that yields the highest accuracy for all features dataset is: <br>
optimizer = rmsprop, activation = sigmoid, batch_size = 30, epoch = 7000, regularizer = 0.001
which yields in accuracy = 0.72254 and out sample accuracy = 0.72308 <br>
<br>
