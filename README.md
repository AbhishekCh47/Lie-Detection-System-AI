# Lie Detection System

## Overview
The Lie Detection System is an AI project that aims to determine the truth value of responses during interviews by analyzing features related to eye movements and facial expressions. The project leverages neural networks optimized using genetic algorithms and hill climbing strategies.

## Project Goals
- Improve the accuracy and speed of the neural network beyond standard backpropagation techniques.
- Utilize a dataset of features captured during interviews to train the neural network.
- Predict the truth value of a person's answer based on the features.

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn

## Approach
1. **Data Preprocessing**:
   - Load and preprocess the dataset containing eye and facial movement features.
   - Split the dataset into training and testing sets.

2. **Neural Network Implementation**:
   - Implement a neural network class that includes methods for:
     - Forward propagation.
     - Accuracy and score calculation.
     - Activation functions (sigmoid and ReLU).
   
3. **Genetic Algorithm**:
   - Create a class to implement the genetic algorithm:
     - Crossover and mutation strategies to optimize neural network weights and biases.
     - Maintain a population of neural networks and evaluate their performance.

4. **Hill Climbing Optimization**:
   - Enhance the neural network's performance using hill climbing techniques.
   - Update weights and biases iteratively to achieve better accuracy.

## Code Structure
### Main Classes
- **Network**: Represents the neural network and includes methods for feedforward propagation, scoring, and accuracy calculation
- **NNGeneticAlgo**: Implements the genetic algorithm for optimizing the neural network.

### Main Functions
- `main()`: The main function that executes the data loading, preprocessing, neural network training, and optimization processes.

## How to Run
1. Ensure you have Python installed with the required packages:
   ```bash
   pip install numpy pandas scikit-learn
   ```
2. Place the dataset file `EYES.csv` in the same directory as the code.
3. Run the script:
   ```bash
   python lie_detection_system.py
   ```

## Expected Output
- The system will output the current iteration, time taken, and the accuracy of the top-performing neural network.

## Conclusion
This project demonstrates an effective approach to lie detection using advanced optimization techniques on neural networks, significantly improving accuracy and speed compared to traditional methods.
