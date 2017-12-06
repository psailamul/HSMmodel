# Config
import os


def get_host_path(HOST=False, PATH=True):
    import socket
    if socket.gethostname() =='x7' :
        host='x7'; path='/home/pachaya/Allen_Brain_Observatory/'
    elif socket.gethostname() =='g13':
        host='g13'; path= '/home/pachaya/Allen_Brain_Observatory/'
    elif socket.gethostname() =='x8' :
        host='x8'; path='/home/pachaya/Allen_Brain_Observatory/'
    elif socket.gethostname() =='x9' :
        host='x9'; path='/home/drew/Documents/Allen_Brain_Observatory/'
    else:
        raise Exception('Unknown Host : Please add your directory at get_host_path()')
    if(HOST&PATH):
        return host, path
    elif(HOST):
        return host
    else:
        return path

        
