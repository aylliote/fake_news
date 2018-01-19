## fake_news

My solution to the fake_news challenge of the Saclay M2 DataCamp 17/18 (https://www.ramp.studio/problems)

I got 1st of the competitive phase of the challenge (smoothed accuracy : 0.471)

## the problem
Train and test datasets are available at : https://github.com/ramp-kits/fake_news/tree/master/data

The goal is to develop a prediction model able to identify which news is fake.

The data is from http://www.politifact.com. The input contains of short statements of public figures (and sometimes anomymous bloggers), plus some metadata. The output is a truth level, judged by journalists at Politifact. They use six truth levels which are coded into integers to obtain an ordinal regression problem:

0: 'Pants on Fire!'
1: 'False'
2: 'Mostly False'
3: 'Half-True'
4: 'Mostly True'
5: 'True'

The aim is to classify each statement (+ metadata) into one of the categories.


# The problem is quite difficult for two reasons : 
- The dataset only includes the short headlines, not the full article. In my opinion lies are not so correlated to the semantic/syntaxic/stylistic/words construction of the statement, rather than the knowledge of the actual context of what it deals with. It was not allowed to use any external data for that challenge.
- The goal is not to simply determine weither a statement is fake or not, but to classify it within one of the 6 levels of truthness, making the model able to distinguish 'False', 'Mostly false, and 'Half true' for instance.

## Feature extraction

I construct the tfidf vectors obtained upon text, tfidf vectors obtained upon pos-tags (with some limitations on vocab size and up to bigrams), which leads to some very-sparse features.
I then construct some ratios with the meta-features. The strategy is to calculate the ratios n_fake_statements/n_total_statement for a given subject, editor, journalist, etc ...
The ratios are then smoothed using Bayesian inference to get some more fair estimates. For instance, I'd say an editor who has published 45 fake articles over 50 total articles is more likely to publish a new fake article than an editor who publish 2 fake articles out of 2.
This is done by fitting beta distributions on the data, see http://varianceexplained.org/r/empirical_bayes_baseball/ .

This allows me to avoid another sparse (one hot encoded) representation of the meta features. (keeping the dimension of the final datasets reduced)

Those two kinds of features are merged, along with a few other customs like the count of undefined articles, the count of numbers, ...






