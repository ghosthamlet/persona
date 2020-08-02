
import sys
import yaml


config_file = sys.argv[1]
args = yaml.load(open(config_file), Loader=yaml.FullLoader)

ret = []
for k, v in args.items():
    ret.append(k + '=' + str(v))

print(' '.join(ret))
