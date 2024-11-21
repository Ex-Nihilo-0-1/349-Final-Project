# final project proposal
- What task will you address, and why is it interesting? 

Predicting and verifying the result of 2024's general election using longitudinal socia-economical data from all 3143 counties in the united states. This is interesting because of the following. (1) It is about current events and so obviously to see what comes out of our model would be very fascinating. This also means that there will be a good audience for our project. (2) Readily available are high quality, vetted data coming from various government institutions (see below), which guarantees that the pronouncements of our models will be high quality (as opposed to "garbage in, garbage out"). (3) It appears to be a scientifically sound application of machine learning (compare using machine learning to predict earthquakes) (4) Election prediction seems to have been a hot topic among machine learning engineers, meaning we can build on experience of others and allow us to build something really nice. (5) US politics is perpetually a hot topic, and since the political institution will operate on unchanging principles, we can play with the model we built this time in the next election and ones after that and it'll still be fun.

- How will you acquire your data? This element is intended to serve as a check
that your project is doable -- so if you plan to collect a new data set (which I discourage), be
as specific as possible.

We will use existing datasets from government institutions such as the USDA and the Bureau of Labor Statistics. By combining these datasets, we aim to create a comprehensive dataset with multiple dimensions, capturing various socio-economic indicators for each county. We expect these data to be vetted by trained professional from these institutions such that they are of high quality.

- Which features/attributes will you use for your task?

We will use factors that intuitively matter for an election. For example, we will consider socio-economic indicators at the county level, such as unemployment rates, median household income, education levels, population demographics (age, race, gender), housing prices. We might also consider past winning candidates' party, whether it was a close call/decisive win, etc. A certain consumer cultural factors such as gun sales, what types of products that the local people spend the most/least on, may also be included in what our features will be based on.


- What will your initial approach be? What data pre-processing will you do, which
machine learning techniques (decision trees, KNN, K-Means, Gaussian mixture models, etc.)
will you use, and how will you evaluate your success (Note: you must use a quantitative
metric)? Generally, you will likely use mean-squared error for regression tasks and
precision-recall for classification tasks. Think about how you will organize your model
outputs to calculate these metrics. 


For data pre-Processing, we will clean and normalize the datasets we acquired to ensure consistency and handle missing values. We expect this to be not a very daunting task since we'll use our data from a reputable source. Categorical data like county names will be encoded. We will use KNN to cluster values of labels. We will also perform feature selection to identify the most influential attributes. 

For machine learning techniques, we will be mainly using two techniques: Decision Trees and MLP(muti-layer perceptron). Decision tree will help us extract the important features from the datasets, and those features will then be fed into the MLP. The MLP will then sequentially learn from those inputs from the decision trees, and give an output for a county, predicting its chances of voting blue or red. Finally, we will then combine these country prediction by states, which will result in an election prediction map by states.

For evaluation metrics we will evaluate our models using metrics such as precision, recall, and F1-score. These metrics will help us understand the model’s ability to correctly classify information. We will compare the model's predictions to actual election results to assess its performance.
