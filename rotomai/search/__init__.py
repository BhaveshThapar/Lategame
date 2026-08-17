"""R-PREDICT (Lever 11): depth-limited search / opponent-modeling on the GREEN base.

The forward model is the vendored Showdown simulator's serialize/fork/step primitive
(``search.fidelity`` validates it; ``search.forward`` wraps it). Search itself is a
test-time procedure layered on the frozen offline-RL checkpoint + value head.
"""
