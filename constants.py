
## stop words
usefull_stopwords = ['very', 'too', 'such', 'only', 'against', 'not', 'nor', 'no', 'most', 'more' ]
english_stopwords = set([w for w in stopwords.words('english') if w not in usefull_stopwords])

## meta features
meta_features = ['source_reliable', 'journalist_likes_truth', 'editor_likes_truth', 
            'job_prone_truth', 'subject_prone_truth', 'state_prone_truth']

## parameters of Beta distributions for the bayesian smoothing
BAYESIAN_PARAMETERS = {ALPHA_SOURCE : 2.761123
                        BETA_SOURCE : 3.628518

                        ALPHA_JOURNALIST : 5.584136
                        BETA_JOURNALIST : 8.167342

                        ALPHA_EDITOR : 7.795503
                        BETA_EDITOR : 10.641396

                        ALPHA_JOB : 2.482408
                        BETA_JOB : 3.815495

                        ALPHA_SUBJECT : 15.24234
                        BETA_SUBJECT : 19.17984

                        ALPHA_STATE : 11.81635
                        BETA_STATE : 15.61433}