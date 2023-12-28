# hog-strategy
My strategy for the hog project in CS61a :)
The actual hog code is hidden as to give away any of the code for the project, but I thought I'd share my code anyway.
I created a feed forward neural net to determine the number of rolls the strategy should take, and utilized the Nerual Evolution of Augmented Topologies algorithm to optimize. 

I gave the neural network 5 inputs: Player score, Opponent score, Player distance to goal, Opponent distance to goal, and Points gained if you roll zero (labeled Boar + Sus Score)

The outputs were the number of rolls the strategy would decide to take, and the argmax of these output nodes decided that number of rolls.

I decided to evaluate fitness as the win percentage against my non-neural network approach to a final strategy--which entailled checking first to see if rolling 0 dice would lead to a victory, then seeing if this point gain was greater than the average points gaineed from 6 rolls. This strategy had around a 70% win percentage against a "always roll 6" strategy.

The end result of the model had a 60% win rate against my final strategy, which I consider to be a pretty big success! Despite the 5 inputs, the model only looks at the opponent score, boar + sus score, and player distance to goal to decide how many dice to roll. The model then chooses between 0, 6 or 2 dice to roll.

While my code does include packages/imports, a non-import implementation of this code could be done by simply hard-coding the neural network that describes the strategy. Since each perceptron can be calculated as sigmoid(Wx + b), the acutal calculations done by the neural network are simple and easy to implement without any external libraries. 
