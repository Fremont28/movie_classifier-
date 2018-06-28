# movie_classifier-
Using feature scaling to classify movie ratings 

We are interested in classifying whether a movie received an above or below average rating based on its budget, popularity, and runtime. 

We will use feature scaling to rescale the features on a -1 to 1 scale. Since the majority of machine learning algorithms use Euclidean distance between two data points in computation it is important that the features are on a similar scale. If not, features of higher magnitudes will have greater importance than those of low magnitude in a classification or regression problem. 

We will in turn use linear discriminant analysis (LDA) and principle component analysis (PCA) to reduce the dimensionality of our movie dataset. Both dimensionality reducing techniques “analyze” the variances of the different values in the movie dataset. 

A random forest classifier will then be used to classify whether the movie scored above or below the average rating (for all movies in the dataset). Overall, the random forest model trained on the LDA scaled dataset slightly outperformed the PCA scaled dataset (96.0% accuracy compared to 93.2% accuracy). 



