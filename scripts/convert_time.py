import sys

time=sys.argv[1]

mins=int(time.split(":")[0])
secs=int(time.split(":")[1])

decs=secs*100/60

time_dec=mins+decs/100

print("%.2f" % time_dec)
