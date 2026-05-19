import json
import time

class WeakSupervisionEngine:
    def __init__(self, action_space: list):
        """
        Stage 4: Automated Action Space & Weak Supervision.
        Initializes the action weights based on the automatically extracted 
        action space from the NLP ingestion pipeline.
        """
        self.action_space = action_space
        # Initialize uniform confidence scores for all actions
        self.action_weights = {action: 1.0 for action in action_space}
        self.history = []

    def submit_retrospective_outcome(self, node: str, action_taken: str, initial_threat: float, realized_outcome: float):
        """
        Missing Element B: Feedback Webhook.
        External clients report the real-world outcome of the LLM's recommendation.
        If the realized threat was lower than expected, the action's weight increases.
        
        Optimized with L2 weight regularization decay and saturation bounds to prevent divergence.
        """
        print(f"\n[Supervision] Received Retrospective Outcome Webhook for '{node}':")
        if action_taken not in self.action_weights:
            print(f"[Supervision WARN] Action '{action_taken}' not in known action space. Ignoring.")
            return

        # Calculate effectiveness (did the threat actually decrease?)
        improvement = initial_threat - realized_outcome
        
        # 1. Decay all weights toward the baseline of 1.0 (L2 Regularization / Weight Decay)
        decay_rate = 0.05
        for act in self.action_weights:
            # Shift slightly back toward 1.0
            self.action_weights[act] = self.action_weights[act] + decay_rate * (1.0 - self.action_weights[act])
            
        # 2. Update learning weight for target action
        learning_rate = 0.1
        if improvement > 0:
            # Action was successful, increase confidence
            self.action_weights[action_taken] += (improvement * learning_rate)
            print(f" -> SUCCESS: Threat reduced by {improvement:.2f}.")
        else:
            # Action failed, decrease confidence
            self.action_weights[action_taken] -= (abs(improvement) * learning_rate)
            print(f" -> FAILURE: Threat increased by {abs(improvement):.2f}.")
            
        # 3. Apply Saturation Bounds [0.1, 5.0] to prevent unbounded divergence
        for act in self.action_weights:
            self.action_weights[act] = max(0.1, min(5.0, self.action_weights[act]))

        print(f" -> Learned Model Update: Confidence in '{action_taken}' adjusted to {self.action_weights[action_taken]:.3f}.")
        
        self.history.append({
            "timestamp": time.time(),
            "node": node,
            "action": action_taken,
            "improvement": improvement
        })

    def get_action_confidence(self):
        """
        Returns the sorted dictionary of normalized action confidence scores.
        Uses a Softmax-like normalization function to present bounded relative confidence ratios.
        """
        import numpy as np
        # Convert weights to exponential scale to obtain a relative probability distribution
        weights = np.array([self.action_weights[a] for a in self.action_space], dtype=np.float64)
        exp_w = np.exp(weights - np.max(weights)) # Subtraction for numerical stability
        softmax_probs = exp_w / np.sum(exp_w)
        
        normalized_weights = {
            self.action_space[i]: float(round(softmax_probs[i], 4))
            for i in range(len(self.action_space))
        }
        return dict(sorted(normalized_weights.items(), key=lambda item: item[1], reverse=True))

if __name__ == "__main__":
    # Simulate receiving the action space extracted from Stage 1
    mock_action_space = ["increase_local_inventory", "forward_contracts"]
    supervision_engine = WeakSupervisionEngine(mock_action_space)
    
    print("\n--- Initial Action Confidence Weights ---")
    print(json.dumps(supervision_engine.get_action_confidence(), indent=2))
    
    # Client 1 took advice, but it didn't work well
    supervision_engine.submit_retrospective_outcome(
        node="AUTOMOTIVEPLANT",
        action_taken="forward_contracts",
        initial_threat=2.70,
        realized_outcome=3.00  # Threat actually got worse!
    )
    
    # Client 2 took advice, and it worked perfectly
    supervision_engine.submit_retrospective_outcome(
        node="BATTERYFACTORY",
        action_taken="increase_local_inventory",
        initial_threat=1.80,
        realized_outcome=0.50  # Threat neutralized
    )
    
    print("\n--- Updated Learned Action Confidence Weights ---")
    print(json.dumps(supervision_engine.get_action_confidence(), indent=2))
