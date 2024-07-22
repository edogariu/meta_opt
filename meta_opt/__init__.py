from jax import random
if not hasattr(random, 'KeyArray'):  # proactively fix the classic deprecation bug we all know and hate
    random.KeyArray = random.PRNGKey