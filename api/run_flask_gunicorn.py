from flask import Flask

from api.run_flask import FlaskConfig, RunFlaskCommand
from Connect4Helpers.agents.mcts_agent import MCTSAgent

# Create the command
command = RunFlaskCommand(
    FlaskConfig(),
    MinimaxAgent(max_depth=3, is_agent1=False)
)

# IMPORTANT:
# This must be a module-level variable
app: Flask = command.run(start=False)
