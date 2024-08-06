import sys
from datetime import datetime

def progress_bar(total_iter, per_iter_width, toolbar_ind):
    # update the bar
    if (toolbar_ind)%per_iter_width == 0:
        now = datetime.now()
        dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
        sys.stdout.write("\r[" + '-'*(toolbar_ind//per_iter_width) + "%d/%d] "%(toolbar_ind+1,total_iter) + dt_string)
        sys.stdout.flush()
    if total_iter == toolbar_ind+1:
        now = datetime.now()
        dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
        sys.stdout.write("\r[" + '-'*(toolbar_ind//per_iter_width) + "%d/%d] "%(toolbar_ind+1,total_iter) + dt_string + '\n')
        sys.stdout.flush()

def convert_to_preferred_format(sec):
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60

    return "%02d:%02d:%02d" % (hour, min, sec)
