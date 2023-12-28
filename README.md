# hog-strategy
My strategy for the hog project in CS61a :)
The actual hog code is hidden as to give away any of the code for the project, but I thought I'd share my code anyway.
I created a feed forward neural net to determine the number of rolls the strategy should take, and ustilized the Nerual Evolution of Augmented Topologies algorithm to optimize. 
- I gave the neural network 5 inputs: Player score, Opponent score, Player distance to goal, Opponent distance to goal, and Points gained if you roll zero (labeled Boar + Sus Score)
- The outputs were the number of rolls the strategy would decide to take, and the argmax of these output nodes decided that number of rolls.
- I decided to evaluate fitness as the win percentage against my non-neural network approach to a final strategy--which entailled checking first to see if rolling 0 dice would lead to a victory, then seeing if this point gain was greater than the average points gaineed from 6 rolls. This strategy had around a 70% win percentage against a "always roll 6" strategy.
