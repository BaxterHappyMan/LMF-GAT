def write_logs(msg, path):
    with open(path, 'a') as fp:
        fp.write(msg + '\n')

def write_loss(loss_list, path):
    with open(path, 'a') as fp:
        for item in loss_list:
            fp.write(str(item) + '\n')