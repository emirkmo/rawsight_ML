# Collaborative Filtering Algorithm

Learn both feature vector X and user parameter (linear regression) vector W and b.
Users (samples) that rated, i.e. have a parameter for a given feature, are kept track of in
an binar matrix R. Matrix Y are the ratings. Features X and parameters W and b must be learned collaboratively.

Features = X
User pars = w, b
R = mapping between users and movie ratings
Y = movie ratings

Y(movie, user) = R(movie, user) * (w(user) . x(movie) + b(user))
