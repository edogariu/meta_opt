
class bcolors:  # for printing pretty colors :)
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# from algorithmic_efficiency.logger_utils import _get_utilization
# def get_total_mem_usg():
#     utilization = _get_utilization()
#     ram_usage = utilization['mem.used']  # add ram usage
    
#     # add GPU memory usage
#     gpu_usage = 0
#     for k, v in utilization.items():
#         if 'gpu' in k and 'mem.used' in k: gpu_usage += v
#     return ram_usage + gpu_usage
