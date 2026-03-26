"""Settings for route evaluation inside each supercluster."""

# Choose from: "soft", "hard"
ROUTING_VARIANT = "soft"

# Choose from: "exact", "heuristic", "hybrid"
ROUTING_SOLVER = "exact"

# In hybrid mode, use exact only up to this many customers.
# Larger superclusters use the heuristic solver.
HYBRID_EXACT_MAX_CUSTOMERS = 45