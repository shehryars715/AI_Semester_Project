from flask import Flask

from api.run_flask import FlaskConfig, RunFlaskCommand
from Connect4Helpers.agents.mcts_agent import MCTSAgent

# Create the command
command = RunFlaskCommand(
    FlaskConfig(),
    MCTSAgent()
)

# IMPORTANT:
# This must be a module-level variable
app: Flask = command.run(start=False)
