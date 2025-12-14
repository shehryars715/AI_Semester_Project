import numpy as np
from tqdm import tqdm
from Connect4Helpers.agents.mcts_agent import MCTSAgent
from Connect4Helpers.interfaces.board import Board
from Connect4Helpers.constants.constants import COLUMN_COUNT
from Connect4Helpers.interfaces.mcts_interface import Connect4Tree
import time

def generate_training_data(num_games=1000, num_rollouts=100):
    """
    Generate training data from MCTS self-play games.
    Args:
        num_games: Number of games to generate
        num_rollouts: Number of MCTS rollouts per move (lower = faster, less accurate)
    Returns: (boards, policies, outcomes)
    """
    boards = []
    policies = []
    outcomes = []
    
    agent = MCTSAgent(simulation_time=999.0, show_pbar=False)  # High timeout, we'll use rollout limit
    
    for game_idx in tqdm(range(num_games), desc="Generating games"):
        board = Board(board=None, turn=0)  # Fixed: provide required arguments
        game_boards = []
        game_policies = []
        game_turns = []
        
        while True:
            # Check if game is over
            if board.winning_move(1) or board.winning_move(2) or board.tie():
                break
            
            # Wrap board for MCTS
            tree_node = Connect4Tree(board.board.copy(), turn=board.turn)
            
            # Get MCTS policy (visit counts as policy)
            policy = np.zeros(COLUMN_COUNT)
            
            # Run MCTS rollouts - FIXED NUMBER for speed
            for _ in range(num_rollouts):
                agent.tree.do_rollout(tree_node)
            
            # Extract visit counts as policy
            policy_dict = agent.tree.get_policy(tree_node, return_dict=True)
            for col in board.action_indices:
                policy[col] = policy_dict.get(col, 0)
            
            # Normalize policy
            if policy.sum() > 0:
                policy = policy / policy.sum()
            
            # Store state
            game_boards.append(board.board.copy())
            game_policies.append(policy.copy())
            game_turns.append(board.turn)
            
            # Make move using the MCTS tree's choose method
            chosen_node = agent.tree.choose(tree_node)
            col = chosen_node.last_move
            
            # Drop the piece
            row = board.get_next_open_row(col)
            if row is not None:
                board.drop_piece(row, col)
        
        # Determine outcome
        winner = None
        if board.winning_move(1):
            winner = 0  # Player 0 (piece 1) won
        elif board.winning_move(2):
            winner = 1  # Player 1 (piece 2) won
        else:
            winner = None  # Draw
        
        # Assign outcomes from perspective of each player
        for game_board, game_policy, turn in zip(game_boards, game_policies, game_turns):
            if winner is None:  # Draw
                outcome = 0.5
            elif winner == turn:  # This player won
                outcome = 1.0
            else:  # This player lost
                outcome = 0.0
            
            boards.append(game_board)
            policies.append(game_policy)
            outcomes.append(outcome)
    
    return np.array(boards), np.array(policies), np.array(outcomes)


if __name__ == "__main__":
    import os
    
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    print("Generating training data...")
    print("Using 50 rollouts per move for speed (increase for better quality)")
    
    # Fast generation: 50 rollouts per move
    train_boards, train_policies, train_outcomes = generate_training_data(
        num_games=10,  # More games
        num_rollouts=50  # Fewer rollouts = faster
    )
    
    print("Generating test data...")
    test_boards, test_policies, test_outcomes = generate_training_data(
        num_games=2,
        num_rollouts=50
    )
    
    # Save in the format expected by training script
    # Use object arrays to handle different shapes
    train_data = np.empty(3, dtype=object)
    train_data[0] = train_boards
    train_data[1] = train_policies
    train_data[2] = train_outcomes
    
    test_data = np.empty(3, dtype=object)
    test_data[0] = test_boards
    test_data[1] = test_policies
    test_data[2] = test_outcomes
    
    np.save("./data/training_1a.npy", train_data)
    np.save("./data/training_1b.npy", test_data)
    
    print(f"Training data: {len(train_outcomes)} samples")
    print(f"Test data: {len(test_outcomes)} samples")
    print("Data saved to ./data/")