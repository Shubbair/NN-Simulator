# AI Modeling Simulator (NN Simulator)

NN Simulator is a lightweight design and brainstorming tool for students and researchers who want to explore neural network ideas, test architectural logic, and quickly understand whether a model structure is conceptually valid before moving to deeper implementation.

The project focuses on helping users:

- build neural network architectures visually
- inspect layer connections and shapes
- validate common design mistakes early
- review training readiness for valid model structures
- export architecture ideas into PyTorch and TensorFlow code

## Testing

Run the validation and debugging checks with:

`npm test -- --test-reporter=spec`

Current test coverage focuses on:

- validation of bad layer arrangements
- validation of invalid skip direction
- validation of a valid Conv → Flatten → Linear flow
- checking training readiness for a valid architecture

This project is intended as a research and learning tool, not as a full production ML training platform.
