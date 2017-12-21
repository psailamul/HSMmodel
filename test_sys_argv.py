import sys
print "This is the name of the script: ", sys.argv[0]
print "Number of arguments: ", len(sys.argv)
print "The arguments are: " , str(sys.argv)
for i in range(len(sys.argv)):
    print "sys.argv[%g] is %s"%(i,sys.argv[i])
