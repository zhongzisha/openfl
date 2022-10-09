import sys

split_num = sys.argv[1]


filename = '/data/zhongz2/openfl/openfl-workspace/torch_histo_mtl_v2/plan/plan.yaml'
with open(filename, 'r') as fp:
    lines = fp.readlines()

line_ind = -1
for line_ind, line in enumerate(lines):
    if 'split_num          ' in line:
        break

if 0 <= line_ind < len(lines):
    lines[line_ind] = '    split_num          : {}\n'.format(split_num)

    with open(filename, 'w') as fp:
        fp.writelines(lines)

    print('replace the split_num successfully!')
else:
    print('cannot find split_num settings, check the plan.yaml!')