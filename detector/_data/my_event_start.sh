#!/bin/bash

# duration=600
duration=30

# use arrays instead of strings to avoid quote hell
python3 /var/lib/zmeventnotification/bin/my_detection/detect.py $2 ${duration}

# The script needs  to return a 0 for success ( detected) or 1 for failure (not detected)
_RETVAL=1

exit ${_RETVAL}
