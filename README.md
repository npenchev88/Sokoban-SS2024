# Sokoban-SS2024

## Overview
Sokoban-SS2024 is a project aimed at developing solutions for the Sokoban game using two different algorithms: 
Reinforcement Learning (RL) and NeuroEvolution of Augmenting Topologies (NEAT). 
This project leverages external repositories, specifically the OpenAI Gym and the gym-sokoban wrapper, 
to create a platform for experimenting with these algorithms.


## Setup and Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher (TODO)
- pip (Python package installer) (TODO)

### Installation Steps
1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/Sokoban-SS2024.git
   cd Sokoban-SS2024

### Challenges and Solutions

The primary challenge encountered was the backward compatibility issues due to the gym-sokoban repository 
not being maintained for the past five years. This caused issues when trying to use a newer version of the OpenAI Gym. 
To address this, we made modifications to the sokoban_env.py file to ensure compatibility and to hardcode a specific 
level of the game for proof-of-concept (POC) purposes.
### Usage

After setting up the environment, you can run the RL and NEAT algorithm implementations from their respective directories. 
Detailed instructions and scripts for training and evaluating the algorithms are provided within each directory.
Running the RL Algorithm

Navigate to the rl_algorithm directory and follow the instructions in the README.md file located there.
Running the NEAT Algorithm

Navigate to the neat_algorithm directory and follow the instructions in the README.md file located there.
Contributing

Contributions to this project are welcome. Please open an issue or submit a pull request with your changes. Ensure your code follows the project's coding standards and includes appropriate tests.
License

This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgements

    OpenAI Gym
    gym-sokoban

### Contact

For any questions or inquiries, please contact your-email@example.com.