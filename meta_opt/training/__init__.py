import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'