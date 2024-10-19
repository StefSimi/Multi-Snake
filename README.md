# Multi-Snake

## About

An project that implements feed-forward neural networks in a genetic alogorithm, for a multi-agent Snake-like game to determine the best strategy for long-term survival

## Game features

The game is a complex version of the popular Snake game, with the following modifications:<br>
-The food can be poisonous, reducing a snake's size instead of increasing it;<br>
-A snake's size, speed and vision decreases over time; <br>
-A snake can eat another snake if one is bigger by a certain percentage than the other; <br>

## AI features
-If a snake detects a piece of food within its sight radius, it will calculate the best way to reach it using the A* algorithm
-The environmental information of each snake is being fed into a fast forward neural network, to calculate the best move to ensure survival at each step.
-At the end of each generation, the best genes are chosen to reproduce;
-Mutations are introduced into new generations to prevent stagnation

