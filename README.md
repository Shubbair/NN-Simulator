# AI Modeling Simulator (NN Simulator)

Demo video: [demo.mp4](demo.gif)

NN Simulator is a lightweight design and brainstorming tool for students and researchers who want to explore neural network ideas, test architectural logic, and quickly understand whether a model structure is conceptually valid before moving to deeper implementation.

The project focuses on helping users:

- build neural network architectures visually
- inspect layer connections and shapes
- validate common design mistakes early
- review training readiness for valid model structures
- export architecture ideas into PyTorch and TensorFlow code

## Analysis Summary

The Analysis tab provides quick, educational feedback about the likely behavior of a model before training begins. It is intended as a design aid, not a full training diagnosis.

### Gradient Flow Analysis

This part checks whether the architecture is likely to maintain stable signal propagation through depth.

It focuses on practical questions such as:

- Is the network too deep to train reliably?
- Does the design include residual or skip-style flow that can help gradients move through the network?
- Are normalization layers present to improve training stability?

In simple terms, deeper models can struggle with gradient flow, so the system highlights depth and stability concerns early.

### Regularization Health

This part estimates whether the model includes basic safeguards against overfitting and structural problems.

It looks for signs such as:

- dropout to reduce overfitting risk
- normalization to improve stability and convergence
- obvious structural mistakes that should be corrected before training

The goal is to help users reason about whether a design is likely to be stable and trainable, even before actual experiments begin.

### Why these checks are useful

These signals help users quickly answer questions like:

- Is this architecture likely to be stable?
- Am I missing a normalizing layer or residual-style path?
- Is the model too deep or too weakly regularized?
- Does the graph have obvious issues that should be fixed first?

The analysis is meant to support better architecture decisions and help researchers learn from model structure, not to replace real training metrics.

## Testing

Run the validation and debugging checks with:

`npm test -- --test-reporter=spec`

Current test coverage focuses on:

- validation of bad layer arrangements
- validation of invalid skip direction
- validation of a valid Conv → Flatten → Linear flow
- checking training readiness for a valid architecture

This project is intended as a research and learning tool, not as a full production ML training platform.
